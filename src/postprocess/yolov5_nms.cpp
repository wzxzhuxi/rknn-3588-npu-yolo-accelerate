// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "yolov5_nms.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
namespace yolov5
{

    // COCO 80 类标签 (硬编码,跟 model 训练时的类别一致)
    static const char *labels[OBJ_CLASS_NUM] = {
        "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
        "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
        "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
        "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "};

    const int anchor0[6] = {10, 13, 16, 30, 33, 23};
    const int anchor1[6] = {30, 61, 62, 45, 59, 119};
    const int anchor2[6] = {116, 90, 156, 198, 373, 326};

    inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

    static float
    CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                     float ymax1)
    {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    static int
    nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
        int filterId, float threshold)
    {
        for (int i = 0; i < validCount; ++i)
        {
            if (order[i] == -1 || classIds[i] != filterId)
            {
                continue;
            }
            int n = order[i];
            for (int j = i + 1; j < validCount; ++j)
            {
                int m = order[j];
                if (m == -1 || classIds[i] != filterId)
                {
                    continue;
                }
                float xmin0 = outputLocations[n * 4 + 0];
                float ymin0 = outputLocations[n * 4 + 1];
                float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
                float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

                float xmin1 = outputLocations[m * 4 + 0];
                float ymin1 = outputLocations[m * 4 + 1];
                float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
                float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

                float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold)
                {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }

    static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
    {
        float key;
        int key_index;
        int low = left;
        int high = right;
        if (left < right)
        {
            key_index = indices[left];
            key = input[left];
            while (low < high)
            {
                while (low < high && input[high] <= key)
                {
                    high--;
                }
                input[low] = input[high];
                indices[low] = indices[high];
                while (low < high && input[low] >= key)
                {
                    low++;
                }
                input[high] = input[low];
                indices[high] = indices[low];
            }
            input[low] = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }

    static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

    static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

    inline static int32_t __clip(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        int8_t res = (int8_t)__clip(dst_val, -128, 127);
        return res;
    }

    static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

    // slow path: 单个 grid 位置确实超阈值时的完整处理 (dequant 4 box + class argmax + push)
    // 抽出来是为了 NEON fast-skip 路径里逐 lane 调用, 跟原本 scalar 路径完全等价
    static inline void handle_hit(int8_t *input, int *anchor, int grid_h, int grid_w, int stride,
                                  int a, int i, int j, int grid_len, int8_t thres_i8,
                                  int32_t zp, float scale,
                                  std::vector<float> &boxes, std::vector<float> &objProbs,
                                  std::vector<int> &classId, int &validCount)
    {
        int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
        int8_t *in_ptr = input + offset;
        int8_t box_confidence = in_ptr[4 * grid_len];
        float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
        float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
        float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
        float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
        box_x = (box_x + j) * (float)stride;
        box_y = (box_y + i) * (float)stride;
        box_w = box_w * box_w * (float)anchor[a * 2];
        box_h = box_h * box_h * (float)anchor[a * 2 + 1];
        box_x -= (box_w / 2.0);
        box_y -= (box_h / 2.0);

        int8_t maxClassProbs = in_ptr[5 * grid_len];
        int maxClassId = 0;
        for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
                maxClassId = k;
                maxClassProbs = prob;
            }
        }
        if (maxClassProbs > thres_i8) {
            objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) *
                               (deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
        }
    }

    static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                       std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                       float threshold,
                       int32_t zp, float scale)
    {
        int validCount = 0;
        int grid_len = grid_h * grid_w;
        // 注意: 假设模型导出时已带 sigmoid head (airockchip yolov5s_relu.onnx + vendor 预转
        // 的 .rknn 都是这样). 这里不额外 sigmoid — 如果换无 sigmoid head 的模型, 要在 box
        // 解码那 4 行加 sigmoid 回去.
        // 诊断信号: 如果换模型后每帧 AvgDet=64.00 (OBJ_NUMB_MAX_SIZE 上限) → sigmoid 不匹配.
        int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

#if defined(__ARM_NEON)
        // NEON 化的 fast-skip: 一次 16 个 grid 位置 compare vs threshold.
        // box_confidence 在内存里是连续的 grid_h * grid_w 个 int8 (每 anchor 一段).
        // ~99% 的 grid 位置 < threshold, 跳过它们 = 跳过 16 个 scalar 比较 + 16 次分支.
        // 只有当 16 个里至少 1 个超阈值才走 scalar slow path (handle_hit).
        const int8x16_t thres_vec = vdupq_n_s8(thres_i8);
        for (int a = 0; a < 3; a++) {
            int8_t *conf_base = input + (PROP_BOX_SIZE * a + 4) * grid_len;
            int idx = 0;
            const int total = grid_len;
            for (; idx + 16 <= total; idx += 16) {
                int8x16_t conf = vld1q_s8(conf_base + idx);
                uint8x16_t mask = vcgeq_s8(conf, thres_vec);
                // 16 字节 mask 折叠成两 64 位字, OR 看是否全 0
                uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(mask), 0);
                uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(mask), 1);
                if ((lo | hi) == 0) continue;   // 全部 fail → 跳过 16 个位置

                // 至少 1 个 hit, 把 mask 存到栈上扫
                alignas(16) uint8_t mb[16];
                vst1q_u8(mb, mask);
                for (int k = 0; k < 16; ++k) {
                    if (mb[k]) {
                        int flat = idx + k;
                        handle_hit(input, anchor, grid_h, grid_w, stride,
                                   a, flat / grid_w, flat % grid_w, grid_len, thres_i8,
                                   zp, scale, boxes, objProbs, classId, validCount);
                    }
                }
            }
            // tail (grid_len < 16 的余数 或不能 16 整除)
            for (; idx < total; ++idx) {
                if (conf_base[idx] >= thres_i8) {
                    handle_hit(input, anchor, grid_h, grid_w, stride,
                               a, idx / grid_w, idx % grid_w, grid_len, thres_i8,
                               zp, scale, boxes, objProbs, classId, validCount);
                }
            }
        }
#else
        // Scalar fallback — 跟原版完全等价, 跑非 ARM 平台时启用
        for (int a = 0; a < 3; a++) {
            for (int i = 0; i < grid_h; i++) {
                for (int j = 0; j < grid_w; j++) {
                    int8_t bc = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                    if (bc >= thres_i8) {
                        handle_hit(input, anchor, grid_h, grid_w, stride,
                                   a, i, j, grid_len, thres_i8,
                                   zp, scale, boxes, objProbs, classId, validCount);
                    }
                }
            }
        }
#endif
        return validCount;
    }

    int
    post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, detect_result_group_t *group)
    {
        memset(group, 0, sizeof(detect_result_group_t));

        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int> classId;

        // stride 8
        int stride0 = 8;
        int grid_h0 = model_in_h / stride0;
        int grid_w0 = model_in_w / stride0;
        int validCount0 = 0;
        validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes,
                              objProbs,
                              classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

        // stride 16
        int stride1 = 16;
        int grid_h1 = model_in_h / stride1;
        int grid_w1 = model_in_w / stride1;
        int validCount1 = 0;
        validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes,
                              objProbs,
                              classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

        // stride 32
        int stride2 = 32;
        int grid_h2 = model_in_h / stride2;
        int grid_w2 = model_in_w / stride2;
        int validCount2 = 0;
        validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes,
                              objProbs,
                              classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

        int validCount = validCount0 + validCount1 + validCount2;
        // no object detect
        if (validCount <= 0)
        {
            return 0;
        }

        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i)
        {
            indexArray.push_back(i);
        }

        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));

        for (auto c : class_set)
        {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
        }

        int last_count = 0;
        group->count = 0;
        /* box valid detect target */
        for (int i = 0; i < validCount; ++i)
        {
            if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
            {
                continue;
            }
            int n = indexArray[i];

            float x1 = filterBoxes[n * 4 + 0];
            float y1 = filterBoxes[n * 4 + 1];
            float x2 = x1 + filterBoxes[n * 4 + 2];
            float y2 = y1 + filterBoxes[n * 4 + 3];
            int id = classId[n];
            float obj_conf = objProbs[i];

            group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
            group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
            group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
            group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
            group->results[last_count].prop = obj_conf;
            const char *label = labels[id];
            strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

            // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
            // group->results[last_count].box.top,
            //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
            last_count++;
        }
        group->count = last_count;

        return 0;
    }

    void deinitPostProcess()
    {
        //        for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        //            if (labels[i] != nullptr) {
        //                free(labels[i]);
        //                labels[i] = nullptr;
        //            }
        //        }
    }

}