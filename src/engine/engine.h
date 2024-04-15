// 接口定义

#ifndef RK3588_DEMO_ENGINE_H
#define RK3588_DEMO_ENGINE_H

#include "types/error.h"
#include "types/datatype.h"

#include <vector>
#include <memory>

class NNEngine
{
public:
    // 这里全部使用纯虚函数（=0），作用是将NNEngine定义为一个抽象类，不能实例化，只能作为基类使用
    // 具体实现需要在子类中实现，这里的实现只是为了定义接口
    // 用这种方式实现封装，可以使得不同的引擎的接口一致，方便使用；也可以隐藏不同引擎的实现细节，方便维护
    virtual ~NNEngine(){};                                                                                               // 析构函数
    virtual nn_error_e LoadModelFile(const char *model_file) = 0;                                                        // 加载模型文件，=0表示纯虚函数，必须在子类中实现
    virtual const std::vector<tensor_attr_s> &GetInputShapes() = 0;                                                      // 获取输入张量的形状
    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;                                                     // 获取输出张量的形状
    virtual nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outpus, bool want_float) = 0; // 运行模型
    
};

std::shared_ptr<NNEngine> CreateRKNNEngine(); // 创建RKNN引擎

#endif // RK3588_DEMO_ENGINE_H
