#include "rga_nv12_to_rgb.h"

#include <cstring>

#include "im2d.h"
#include "rga.h"
#include "utils/logging.h"

void nv12_to_rgb_mat_rga(uint8_t *nv12_data,
                         int src_w, int src_h,
                         int src_ws, int src_hs,
                         int dst_w, int dst_h,
                         cv::Mat &rgb_out)
{
    // 分配独立 RGB Mat (连续布局, stride = dst_w * 3). cv::Mat refcount 让数据跨线程入队安全.
    rgb_out.create(dst_h, dst_w, CV_8UC3);

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)nv12_data,
                                              src_w, src_h,
                                              RK_FORMAT_YCbCr_420_SP,
                                              src_ws, src_hs);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)rgb_out.data,
                                              dst_w, dst_h,
                                              RK_FORMAT_RGB_888);

    im_rect empty_rect;
    memset(&empty_rect, 0, sizeof(empty_rect));
    int ret = imcheck(src, dst, empty_rect, empty_rect);
    if (IM_STATUS_NOERROR != ret) {
        NN_LOG_ERROR("nv12_to_rgb_mat_rga imcheck failed: %s", imStrError((IM_STATUS)ret));
        return;
    }

    // 一步 RGA: NV12 → RGB888, 同时做 resize. 跟 vendor demo main_video.cc 同款.
    imresize(src, dst);
}
