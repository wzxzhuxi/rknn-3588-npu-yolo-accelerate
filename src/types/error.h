// 错误码定义

#ifndef RK3588_DEMO_ERROR_H
#define RK3588_DEMO_ERROR_H

typedef enum
{
    NN_SUCCESS = 0,                 // 成功
    NN_LOAD_MODEL_FAIL = -1,        // 加载模型失败
    NN_RKNN_INIT_FAIL = -2,         // rknn初始化失败
    NN_RKNN_QUERY_FAIL = -3,        // rknn查询失败
    NN_RKNN_INPUT_SET_FAIL = -4,    // rknn设置输入数据失败
    NN_RKNN_RUNTIME_ERROR = -5,     // rknn运行时错误
    NN_IO_NUM_NOT_MATCH = -6,       // 输入输出数量不匹配
    NN_RKNN_OUTPUT_GET_FAIL = -7,   // rknn获取输出数据失败
    NN_RKNN_INPUT_ATTR_ERROR = -8,  // rknn输入数据属性错误
    NN_RKNN_OUTPUT_ATTR_ERROR = -9, // rknn输出数据属性错误
    NN_RKNN_MODEL_NOT_LOAD = -10,   // rknn模型未加载
    NN_STOPED = -11,                // 程序已停止
    NN_TIMEOUT = -12,          // 超时
} nn_error_e;

#endif // RK3588_DEMO_ERROR_H
