// 单步 RGA 硬件预处理: NV12 → RGB888 同时做 resize.
// 给 MPP 解码回调用 — 输入是回调里的 NV12 虚地址 (返回后会被复用), 输出是独立 cv::Mat.

#ifndef RK3588_DEMO_RGA_NV12_TO_RGB_H
#define RK3588_DEMO_RGA_NV12_TO_RGB_H

#include <opencv2/core/core.hpp>
#include <cstdint>

// NV12 → dst_w x dst_h RGB888 cv::Mat, 单次 RGA imresize 调用.
// 必须在 MPP 解码回调 (同线程, 同步) 里调用, 因为 nv12_data 是 MPP buffer 借出的虚地址.
// 输出 rgb_out 是独立 cv::Mat (内部 malloc), refcount 共享数据, 可跨线程入队.
//   nv12_data:     MPP NV12 虚地址 (回调内有效)
//   src_w/src_h:   真实尺寸
//   src_ws/src_hs: buffer stride
//   dst_w/dst_h:   目标尺寸 (通常是模型输入, 如 640x640)
void nv12_to_rgb_mat_rga(uint8_t *nv12_data,
                         int src_w, int src_h,
                         int src_ws, int src_hs,
                         int dst_w, int dst_h,
                         cv::Mat &rgb_out);

#endif // RK3588_DEMO_RGA_NV12_TO_RGB_H
