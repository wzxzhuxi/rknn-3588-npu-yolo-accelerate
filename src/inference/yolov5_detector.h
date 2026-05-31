// 单个 YOLOv5 推理实例 — 持有一个 NPU context, 跑零拷贝 DMA-BUF 推理.
// 一个 worker thread 拥有一个 Yolov5Detector, N 个 worker 各跑一份 (见 worker_pool.h).

#ifndef RK3588_DEMO_YOLOV5_DETECTOR_H
#define RK3588_DEMO_YOLOV5_DETECTOR_H

#include <cstdint>
#include <vector>
#include <memory>

#include "types/yolo_types.h"
#include "engine/engine.h"

class Yolov5Detector
{
public:
    Yolov5Detector();
    ~Yolov5Detector();

    nn_error_e LoadModel(const char *model_path);
    // 把这个实例的 NPU context 显式绑到 RK3588 三核之一 (core_id ∈ {0,1,2}).
    // 不绑则 librknnrt 走 RKNN_NPU_CORE_AUTO, 多 context 可能挤同一核, 失去多核加速.
    nn_error_e SetCoreMask(int core_id);

    // 主推理路径: 输入是已经 RGA-resize 好的 640x640 RGB888 数据,
    // 函数内做 NEON XOR-128 (uint8→int8) + rknn_run + NMS 后处理.
    //   rgb_data:    模型期望尺寸的 RGB888 字节, layout HWC
    //   src_w/src_h: 原图尺寸, 用于 postprocess 反算 bbox scale
    //   objects:     out, NMS 后的检测框
    nn_error_e Run(const uint8_t *rgb_data,
                   int src_w, int src_h,
                   std::vector<Detection> &objects);

private:
    tensor_data_s input_tensor_;                 // .data 指向 engine 的零拷贝 input mem
    std::vector<tensor_data_s> output_tensors_;  // .data 指向 engine 的零拷贝 output mems
    std::vector<int32_t> out_zps_;
    std::vector<float>   out_scales_;
    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_YOLOV5_DETECTOR_H
