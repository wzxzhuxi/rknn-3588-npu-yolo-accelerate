// rknn_engine.h的实现

#include "rknn_engine.h"

#include <string.h>

#include "utils/engine_helper.h"
#include "utils/logging.h"

static const int g_max_io_num = 10; // 最大输入输出张量的数量

/**
 * @brief 加载模型文件、初始化rknn context、获取rknn版本信息、获取输入输出张量的信息
 * @param model_file 模型文件路径
 * @return nn_error_e 错误码
 */
nn_error_e RKEngine::LoadModelFile(const char *model_file)
{
    int model_len = 0;                               // 模型文件大小
    auto model = load_model(model_file, &model_len); // 加载模型文件
    if (model == nullptr)
    {
        NN_LOG_ERROR("load model file %s fail!", model_file);
        return NN_LOAD_MODEL_FAIL; // 返回错误码：加载模型文件失败
    }
    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL); // 初始化rknn context
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL; // 返回错误码：初始化rknn context失败
    }
    // 打印初始化成功信息
    NN_LOG_INFO("rknn_init success!");
    ctx_created_ = true;

    // 获取rknn版本信息
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

/**
 * @brief 运行模型，获得推理结果
 * @param inputs 输入张量
 * @param outputs 输出张量
 * @param want_float 是否需要float类型的输出
 * @return nn_error_e 错误码
 */
nn_error_e RKEngine::Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float)
{
    // 检查输入输出张量的数量是否匹配
    if (inputs.size() != input_num_)
    {
        NN_LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputs.size(), input_num_);
        return NN_IO_NUM_NOT_MATCH;
    }
    if (outputs.size() != output_num_)
    {
        NN_LOG_ERROR("outputs num not match! outputs.size()=%ld, output_num_=%d", outputs.size(), output_num_);
        return NN_IO_NUM_NOT_MATCH;
    }

    // 设置rknn inputs
    rknn_input rknn_inputs[g_max_io_num];
    for (int i = 0; i < inputs.size(); i++)
    {
        // 将自定义的tensor_data_s转换为rknn_input
        rknn_inputs[i] = tensor_data_to_rknn_input(inputs[i]);
    }
    int ret = rknn_inputs_set(rknn_ctx_, (uint32_t)inputs.size(), rknn_inputs);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_inputs_set fail! ret=%d", ret);
        return NN_RKNN_INPUT_SET_FAIL;
    }

    // 推理
    NN_LOG_DEBUG("rknn running...");
    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_run fail! ret=%d", ret);
        return NN_RKNN_RUNTIME_ERROR;
    }

    // 获得输出
    rknn_output rknn_outputs[g_max_io_num];
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    for (int i = 0; i < output_num_; ++i)
    {
        rknn_outputs[i].want_float = want_float ? 1 : 0;
    }
    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        NN_LOG_ERROR("rknn_outputs_get fail! ret=%d", ret);
        return NN_RKNN_OUTPUT_GET_FAIL;
    }

    NN_LOG_DEBUG("output num: %d", output_num_);
    // copy rknn outputs to tensor_data_s
    for (int i = 0; i < output_num_; ++i)
    {
        // 将rknn_output转换为自定义的tensor_data_s
        rknn_output_to_tensor_data(rknn_outputs[i], outputs[i]);
        NN_LOG_DEBUG("output[%d] size=%d", i, outputs[i].attr.size);
        free(rknn_outputs[i].buf); // 释放缓存
    }
    return NN_SUCCESS;
}

// 析构函数
RKEngine::~RKEngine()
{
    if (ctx_created_)
    {
        rknn_destroy(rknn_ctx_);
        NN_LOG_INFO("rknn context destroyed!");
    }
}

// 创建RKNN引擎
std::shared_ptr<NNEngine> CreateRKNNEngine()
{
    return std::make_shared<RKEngine>();
}
