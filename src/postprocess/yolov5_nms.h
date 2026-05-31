// YOLOv5 后处理 — int8 网络输出 → box 解码 → 类别 argmax → NMS → Detection 列表.
// 输入是 NPU 写在零拷贝 DMA-BUF 里的三个 anchor 尺度 (80x80 / 40x40 / 20x20) 的 int8 张量,
// 输出是单帧检测框组. NEON 在 process() 里做 fast-skip threshold 扫描.

#ifndef RK3588_DEMO_YOLOV5_NMS_H
#define RK3588_DEMO_YOLOV5_NMS_H

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64   // 单帧最大检测数 — 超过会截断
#define OBJ_CLASS_NUM     80   // COCO 80 类
#define NMS_THRESH        0.45
#define BOX_THRESH        0.45
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

namespace yolov5 {

    typedef struct _BOX_RECT {
        int left;
        int right;
        int top;
        int bottom;
    } BOX_RECT;

    typedef struct __detect_result_t {
        char name[OBJ_NAME_MAX_SIZE];
        BOX_RECT box;
        int id;
        float prop;
    } detect_result_t;

    typedef struct _detect_result_group_t {
        int id;
        int count;
        detect_result_t results[OBJ_NUMB_MAX_SIZE];
    } detect_result_group_t;

    int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                     float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                     std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                     detect_result_group_t *group);

    void deinitPostProcess();
}
#endif // RK3588_DEMO_YOLOV5_NMS_H
