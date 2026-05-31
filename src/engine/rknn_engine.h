// RKNPU2 后端 — NNEngine 的 RK3588 实现.
// 走零拷贝路径: LoadModelFile 里 rknn_create_mem + rknn_set_io_mem 一次性绑定 DMA-BUF,
// 之后 RunZeroCopy 只调 rknn_run, 不再走 rknn_inputs_set/outputs_get.

#ifndef RK3588_DEMO_RKNN_ENGINE_H
#define RK3588_DEMO_RKNN_ENGINE_H

#include "engine.h"

#include <vector>
#include <rknn_api.h>

class RKEngine : public NNEngine
{
public:
    RKEngine() : rknn_ctx_(0), ctx_created_(false), input_num_(0), output_num_(0) {}
    ~RKEngine() override;

    nn_error_e LoadModelFile(const char *model_file) override;
    const std::vector<tensor_attr_s> &GetInputShapes()  override;
    const std::vector<tensor_attr_s> &GetOutputShapes() override;
    nn_error_e SetCoreMask(int core_id) override;

    void *GetInputMemAddr() override;
    void *GetOutputMemAddr(int idx) override;
    nn_error_e RunZeroCopy() override;

private:
    rknn_context rknn_ctx_;
    bool         ctx_created_;
    uint32_t     input_num_;
    uint32_t     output_num_;

    std::vector<tensor_attr_s> in_shapes_;
    std::vector<tensor_attr_s> out_shapes_;

    // NPU 可见 DMA-BUF, LoadModelFile 时一次性 rknn_create_mem + rknn_set_io_mem, 析构释放
    rknn_tensor_mem               *input_mem_ = nullptr;
    std::vector<rknn_tensor_mem *> output_mems_;
};

#endif // RK3588_DEMO_RKNN_ENGINE_H
