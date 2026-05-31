#include "yolov5_detector.h"

#include <chrono>
#include <atomic>
#include <cstring>
#include <ctime>

#include "utils/logging.h"
#include "postprocess/yolov5_nms.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// uint8 RGB → int8 等价于 减 128, 等价于 byte XOR 0x80
//   uint8 0   → int8 -128 (byte 0x80)
//   uint8 128 → int8  0   (byte 0x00)
//   uint8 255 → int8 +127 (byte 0x7F)
// 配合 rknn_tensor_attr.pass_through=1 直接喂 NPU, 省 driver 的 normalize+quant 前置算子.
// 内存带宽 bound, ~0.4ms / 1.17MB. NEON 16 字节并行 + 64 字节展开让 OoO 流水把 LSU 喂满.
static inline void copy_uint8_to_int8_shift128(uint8_t *dst, const uint8_t *src, size_t n)
{
#if defined(__ARM_NEON)
    const uint8x16_t bias = vdupq_n_u8(0x80);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t v0 = vld1q_u8(src + i +  0);
        uint8x16_t v1 = vld1q_u8(src + i + 16);
        uint8x16_t v2 = vld1q_u8(src + i + 32);
        uint8x16_t v3 = vld1q_u8(src + i + 48);
        vst1q_u8(dst + i +  0, veorq_u8(v0, bias));
        vst1q_u8(dst + i + 16, veorq_u8(v1, bias));
        vst1q_u8(dst + i + 32, veorq_u8(v2, bias));
        vst1q_u8(dst + i + 48, veorq_u8(v3, bias));
    }
    for (; i + 16 <= n; i += 16) {
        vst1q_u8(dst + i, veorq_u8(vld1q_u8(src + i), bias));
    }
    for (; i < n; ++i) dst[i] = src[i] ^ 0x80;
#else
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] ^ 0x80;
#endif
}

// 把后处理输出的内部 detect_result_group_t 翻译成对外 Detection 列表
void ConvertDetections(yolov5::detect_result_group_t &det_grp, std::vector<Detection> &objects)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < det_grp.count; i++)
    {
        Detection det;
        det.className = det_grp.results[i].name;
        det.box = cv::Rect(det_grp.results[i].box.left,
                           det_grp.results[i].box.top,
                           det_grp.results[i].box.right - det_grp.results[i].box.left,
                           det_grp.results[i].box.bottom - det_grp.results[i].box.top);
        det.confidence = det_grp.results[i].prop;
        det.class_id = 0;
        det.color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        objects.push_back(det);
    }
}

Yolov5Detector::Yolov5Detector()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
}

// input/output tensors 的 .data 都是 borrowed pointer (指向 engine 的零拷贝 mem),
// engine 析构里会 rknn_destroy_mem 释放, 这里只置空指针.
Yolov5Detector::~Yolov5Detector()
{
    input_tensor_.data = nullptr;
    for (auto &tensor : output_tensors_) {
        tensor.data = nullptr;
    }
}

nn_error_e Yolov5Detector::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS) {
        NN_LOG_ERROR("yolo load model file failed");
        return ret;
    }
    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1) {
        NN_LOG_ERROR("yolo input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);

    // 零拷贝: input_tensor_.data 指向 engine 的 NPU 可见 DMA-BUF.
    // 写这里 → driver 自动 flush cache → rknn_run 直读, 省 rknn_inputs_set 内部 1.17MB 拷贝.
    input_tensor_.data = engine_->GetInputMemAddr();
    if (input_tensor_.data == nullptr) {
        NN_LOG_ERROR("engine returned nullptr input mem — zero-copy not initialized?");
        return NN_RKNN_INIT_FAIL;
    }

    auto output_shapes = engine_->GetOutputShapes();
    for (int i = 0; i < output_shapes.size(); i++) {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims  = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++) {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        if (output_shapes[i].type != NN_TENSOR_INT8) {
            NN_LOG_ERROR("yolo output tensor type is not int8, but %d", output_shapes[i].type);
            return NN_RKNN_OUTPUT_ATTR_ERROR;
        }
        tensor.attr.type  = output_shapes[i].type;
        tensor.attr.index = i;
        tensor.attr.size  = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        // 零拷贝: 不 malloc, 借 engine 的 output DMA-BUF.
        // rknn_run 完成后 driver 已经把 NPU 写的结果 invalidate 进 CPU cache, postprocess 直接读.
        tensor.data = engine_->GetOutputMemAddr(i);
        if (tensor.data == nullptr) {
            NN_LOG_ERROR("engine returned nullptr output mem[%d]", i);
            return NN_RKNN_INIT_FAIL;
        }
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }
    return NN_SUCCESS;
}

nn_error_e Yolov5Detector::SetCoreMask(int core_id)
{
    return engine_->SetCoreMask(core_id);
}

// 阶段计时累计器 — 跨 worker 累加, 每 50 次平均一次, 用于诊断瓶颈
// (memcpy 约 0.4ms, NPU 单帧观察值 67ms = 16.6ms × 4 worker 队列, postprocess 约 0.85ms)
static std::atomic<int>  g_stage_calls{0};
static std::atomic<long> g_t_memcpy_us{0};
static std::atomic<long> g_t_npu_us{0};
static std::atomic<long> g_t_post_us{0};

nn_error_e Yolov5Detector::Run(const uint8_t *rgb_data, int src_w, int src_h,
                               std::vector<Detection> &objects)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    // (1) 写入 NPU 可见输入 mem, 同时 uint8→int8 转换 (配合 pass_through=1)
    copy_uint8_to_int8_shift128(static_cast<uint8_t *>(input_tensor_.data),
                                rgb_data, input_tensor_.attr.size);
    auto t1 = clk::now();

    // (2) NPU 推理 — 仅一次 rknn_run, 无 inputs_set/outputs_get 开销
    nn_error_e run_ret = engine_->RunZeroCopy();
    if (run_ret != NN_SUCCESS) {
        // NPU 失败: output_mem 数据未定义, 跳过 postprocess 直接返回错误.
        // 调用方 (worker) 会 log + 继续下一帧, 保证帧 id 不乱序.
        return run_ret;
    }
    auto t2 = clk::now();

    // (3) 后处理: 用真实原图尺寸反算 bbox scale (跟 RGA 拉伸的反变换)
    const int model_h = static_cast<int>(input_tensor_.attr.dims[1]);
    const int model_w = static_cast<int>(input_tensor_.attr.dims[2]);
    const float scale_w = static_cast<float>(model_w) / static_cast<float>(src_w);
    const float scale_h = static_cast<float>(model_h) / static_cast<float>(src_h);

    yolov5::detect_result_group_t detections;
    yolov5::post_process(
        (int8_t *)output_tensors_[0].data,
        (int8_t *)output_tensors_[1].data,
        (int8_t *)output_tensors_[2].data,
        model_h, model_w,
        BOX_THRESH, NMS_THRESH,
        scale_w, scale_h,
        out_zps_, out_scales_,
        &detections);

    ConvertDetections(detections, objects);
    auto t3 = clk::now();

    g_t_memcpy_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    g_t_npu_us   .fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    g_t_post_us  .fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());
    int n = g_stage_calls.fetch_add(1) + 1;
    if (n % 50 == 0) {
        long memcpy_us = g_t_memcpy_us.exchange(0);
        long npu       = g_t_npu_us   .exchange(0);
        long post      = g_t_post_us  .exchange(0);
        g_stage_calls.store(0);
        NN_LOG_INFO("[STAGE x50] memcpy=%.2fms NPU=%.2fms Post=%.2fms (avg)",
                    memcpy_us / 50.0 / 1000.0, npu / 50.0 / 1000.0, post / 50.0 / 1000.0);
    }
    return NN_SUCCESS;
}
