// 神经网络 engine 抽象接口. 当前唯一实现是 RKNN (RK3588 NPU), 留出虚函数槽位以便将来接其他后端.
// 主路径走零拷贝 API: GetInputMemAddr / GetOutputMemAddr / RunZeroCopy
//   — LoadModelFile 时一次性创建 DMA-BUF + rknn_set_io_mem 绑定,
//     此后 input/output 永驻 NPU 可见内存, 调用方直接读写 virt_addr, 免去 rknn_inputs_set/outputs_get 拷贝.

#ifndef RK3588_DEMO_ENGINE_H
#define RK3588_DEMO_ENGINE_H

#include "types/error_codes.h"
#include "types/tensor_types.h"

#include <vector>
#include <memory>

class NNEngine
{
public:
    virtual ~NNEngine(){}

    // 加载 .rknn 模型 + 初始化 context + 创建零拷贝 I/O DMA-BUF
    virtual nn_error_e LoadModelFile(const char *model_file) = 0;

    virtual const std::vector<tensor_attr_s> &GetInputShapes()  = 0;
    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;

    // 把当前 context 绑到指定 NPU 核 (RK3588: core_id ∈ {0,1,2}).
    // 必须在 LoadModelFile 之后调用. 默认实现是 no-op, 给非 RKNN 后端留口子.
    virtual nn_error_e SetCoreMask(int /*core_id*/) { return NN_SUCCESS; }

    // 零拷贝输入: 返回 NPU 可见 DMA-BUF 的虚地址.
    // 调用方直接写入 INT8 数据 (配合 pass_through=1, driver 不做 normalize/quant).
    virtual void *GetInputMemAddr() { return nullptr; }

    // 零拷贝输出: 第 idx 个 output 的虚地址. postprocess 直接读 int8 dequant.
    virtual void *GetOutputMemAddr(int /*idx*/) { return nullptr; }

    // 零拷贝推理: 仅一次 rknn_run, I/O mem 已经在 LoadModelFile 绑好.
    virtual nn_error_e RunZeroCopy() { return NN_RKNN_RUNTIME_ERROR; }
};

std::shared_ptr<NNEngine> CreateRKNNEngine();

#endif // RK3588_DEMO_ENGINE_H
