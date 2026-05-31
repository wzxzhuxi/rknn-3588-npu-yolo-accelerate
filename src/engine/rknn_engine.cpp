#include "rknn_engine.h"

#include <string.h>

#include "engine/rknn_helpers.h"
#include "utils/logging.h"

// 加载模型 + rknn_init context + 查询 I/O 元数据 + 一次性创建并绑定零拷贝 DMA-BUF
nn_error_e RKEngine::LoadModelFile(const char *model_file)
{
    int model_len = 0;
    auto model = load_model(model_file, &model_len);
    if (model == nullptr) {
        NN_LOG_ERROR("load model file %s fail!", model_file);
        return NN_LOAD_MODEL_FAIL;
    }
    // flag = 0 默认就是 RKNN_FLAG_PRIOR_HIGH (注意它的宏值就是 0, 不写也是 HIGH)
    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL);
    if (ret < 0) {
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL;
    }
    NN_LOG_INFO("rknn_init success!");
    ctx_created_ = true;

    rknn_sdk_version version;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    // 打印rknn版本信息
    NN_LOG_INFO("RKNN API version: %s", version.api_version);
    NN_LOG_INFO("RKNN Driver version: %s", version.drv_version);

    // 获取输入输出个数
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    NN_LOG_INFO("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    // 保存输入输出个数
    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    // 输入属性
    NN_LOG_INFO("input tensors:");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(input_attrs[i]));
        // set input_shapes_
        in_shapes_.push_back(rknn_tensor_attr_convert(input_attrs[i]));
    }

    // 输出属性
    NN_LOG_INFO("output tensors:");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(output_attrs[i]));
        // set output_shapes_
        out_shapes_.push_back(rknn_tensor_attr_convert(output_attrs[i]));
    }

    // 一次性创建 + 绑定 NPU 可见的 I/O DMA-BUF.
    // 之后 worker 不再调 rknn_inputs_set / rknn_outputs_get, 直接读写 mem->virt_addr.
    //
    // 输入: type=INT8 + pass_through=1 让 driver 完全不转换, buf 数据直接喂 NPU.
    //   配合 worker 端的 NEON XOR 0x80 把 uint8 RGB 转 int8.
    //   yolov5s 量化参数 mean=0 / std=255 / zp=-128 / scale≈1/255 时,
    //   driver 的 normalize+quant 等价于 int8 = uint8 - 128, 即 XOR 0x80.
    //   不打 pass_through 会让 driver 在 NPU 上隐式做这一步, 占用 NPU 时间.
    {
        input_attrs[0].type         = RKNN_TENSOR_INT8;
        input_attrs[0].fmt          = RKNN_TENSOR_NHWC;
        input_attrs[0].pass_through = 1;
        input_mem_ = rknn_create_mem(rknn_ctx_, input_attrs[0].size_with_stride);
        if (input_mem_ == nullptr) {
            NN_LOG_ERROR("rknn_create_mem(input, size=%u) returned nullptr", input_attrs[0].size_with_stride);
            return NN_RKNN_INIT_FAIL;
        }
        ret = rknn_set_io_mem(rknn_ctx_, input_mem_, &input_attrs[0]);
        if (ret < 0) {
            NN_LOG_ERROR("rknn_set_io_mem(input) fail! ret=%d", ret);
            return NN_RKNN_INIT_FAIL;
        }
        NN_LOG_INFO("zero-copy input mem: virt=%p fd=%d size=%u",
                    input_mem_->virt_addr, input_mem_->fd, input_attrs[0].size_with_stride);
    }

    // 输出: 保持 model native INT8 NCHW. postprocess 直接 qnt_affine_to_f32 吃 int8, 不要 float.
    output_mems_.resize(output_num_, nullptr);
    for (uint32_t i = 0; i < output_num_; ++i) {
        output_mems_[i] = rknn_create_mem(rknn_ctx_, output_attrs[i].size_with_stride);
        if (output_mems_[i] == nullptr) {
            NN_LOG_ERROR("rknn_create_mem(output[%u], size=%u) returned nullptr",
                         i, output_attrs[i].size_with_stride);
            return NN_RKNN_INIT_FAIL;
        }
        ret = rknn_set_io_mem(rknn_ctx_, output_mems_[i], &output_attrs[i]);
        if (ret < 0) {
            NN_LOG_ERROR("rknn_set_io_mem(output[%u]) fail! ret=%d", i, ret);
            return NN_RKNN_INIT_FAIL;
        }
        NN_LOG_INFO("zero-copy output[%u] mem: virt=%p fd=%d size=%u",
                    i, output_mems_[i]->virt_addr, output_mems_[i]->fd, output_attrs[i].size_with_stride);
    }

    return NN_SUCCESS;
}

void *RKEngine::GetInputMemAddr()
{
    return input_mem_ ? input_mem_->virt_addr : nullptr;
}

void *RKEngine::GetOutputMemAddr(int idx)
{
    if (idx < 0 || idx >= static_cast<int>(output_mems_.size())) return nullptr;
    return output_mems_[idx] ? output_mems_[idx]->virt_addr : nullptr;
}

// 仅一个 rknn_run 调用. driver 默认自动 flush input cache (TO_DEVICE) +
// invalidate output cache (FROM_DEVICE), 不需要手动 rknn_mem_sync
// (只有打开 RKNN_FLAG_DISABLE_FLUSH_* flag 才需要, 我们没用).
nn_error_e RKEngine::RunZeroCopy()
{
    int ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0) {
        NN_LOG_ERROR("rknn_run (zero-copy) fail! ret=%d", ret);
        return NN_RKNN_RUNTIME_ERROR;
    }
    return NN_SUCCESS;
}

// 获取输入张量的形状
const std::vector<tensor_attr_s> &RKEngine::GetInputShapes()
{
    return in_shapes_;
}

// 获取输出张量的形状
const std::vector<tensor_attr_s> &RKEngine::GetOutputShapes()
{
    return out_shapes_;
}

// 把当前 rknn context 显式绑到指定 NPU 核 (RK3588 有 3 个: core0/1/2).
// 不调此函数则走默认 RKNN_NPU_CORE_AUTO 调度, N 个 context 可能挤同一个核, 失去多核加速.
nn_error_e RKEngine::SetCoreMask(int core_id)
{
    if (!ctx_created_) {
        NN_LOG_ERROR("SetCoreMask called before LoadModelFile");
        return NN_RKNN_MODEL_NOT_LOAD;
    }
    rknn_core_mask mask;
    switch (core_id) {
        case 0: mask = RKNN_NPU_CORE_0; break;
        case 1: mask = RKNN_NPU_CORE_1; break;
        case 2: mask = RKNN_NPU_CORE_2; break;
        default:
            NN_LOG_ERROR("SetCoreMask: invalid core_id=%d, expect 0/1/2", core_id);
            return NN_RKNN_INIT_FAIL;
    }
    int ret = rknn_set_core_mask(rknn_ctx_, mask);
    if (ret < 0) {
        NN_LOG_ERROR("rknn_set_core_mask fail! ret=%d core_id=%d", ret, core_id);
        return NN_RKNN_INIT_FAIL;
    }
    NN_LOG_INFO("RKEngine: bound to NPU core %d (mask=%d)", core_id, (int)mask);
    return NN_SUCCESS;
}

// 析构函数: 先释放零拷贝 mems, 再 destroy context (顺序重要 — destroy 后 mem 无法访问)
RKEngine::~RKEngine()
{
    if (ctx_created_)
    {
        if (input_mem_) {
            rknn_destroy_mem(rknn_ctx_, input_mem_);
            input_mem_ = nullptr;
        }
        for (auto *m : output_mems_) {
            if (m) rknn_destroy_mem(rknn_ctx_, m);
        }
        output_mems_.clear();
        rknn_destroy(rknn_ctx_);
        NN_LOG_INFO("rknn context destroyed!");
    }
}

// 创建RKNN引擎
std::shared_ptr<NNEngine> CreateRKNNEngine()
{
    return std::make_shared<RKEngine>();
}
