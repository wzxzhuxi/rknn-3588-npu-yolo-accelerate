// RKNN engine 内部用的小工具: 文件加载 + rknn_tensor_attr ↔ 我们的 tensor_attr_s 互转 + 打印.
// 只给 rknn_engine.cpp 用, 不对外暴露 (header-only static).

#ifndef RK3588_DEMO_RKNN_HELPERS_H
#define RK3588_DEMO_RKNN_HELPERS_H

#include <fstream>
#include <string.h>
#include <vector>

#include <rknn_api.h>

#include "utils/logging.h"
#include "types/tensor_types.h"

/**
 * @brief 加载模型文件
 * @param filename 模型文件路径
 * @param model_size 模型文件大小，会在函数内部赋值
 * @return unsigned char* 模型文件数据
 */
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        NN_LOG_ERROR("fopen %s fail!", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        NN_LOG_ERROR("fread %s fail!", filename);
        free(model);
        return nullptr;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

static void print_tensor_attr(rknn_tensor_attr *attr)
{
    NN_LOG_INFO("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
                "zp=%d, scale=%f",
                attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
                attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
                get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static tensor_layout_e rknn_layout_convert(rknn_tensor_format fmt)
{
    switch (fmt)
    {
    case RKNN_TENSOR_NCHW:
        return NN_TENSOR_NCHW;
    case RKNN_TENSOR_NHWC:
        return NN_TENSOR_NHWC;
    default:
        return NN_TENSOR_OTHER;
    }
}

static tensor_datatype_e rknn_type_convert(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_UINT8:
        return NN_TENSOR_UINT8;
    case RKNN_TENSOR_FLOAT32:
        return NN_TENSOR_FLOAT;
    case RKNN_TENSOR_INT8:
        return NN_TENSOR_INT8;
    case RKNN_TENSOR_FLOAT16:
        return NN_TENSOR_FLOAT16;
    default:
        NN_LOG_ERROR("unsupported rknn type: %d\n", type);
        // exit program
        exit(1);
    }
}

static tensor_attr_s rknn_tensor_attr_convert(const rknn_tensor_attr &attr)
{
    tensor_attr_s shape;
    shape.n_dims = attr.n_dims;
    shape.index = attr.index;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        shape.dims[i] = attr.dims[i];
    }
    shape.size = attr.size;
    shape.n_elems = attr.n_elems;
    // set layout
    shape.layout = rknn_layout_convert(attr.fmt);
    shape.type = rknn_type_convert(attr.type);
    shape.zp = attr.zp;
    shape.scale = attr.scale;
    return shape;
}



#endif // RK3588_DEMO_RKNN_HELPERS_H
