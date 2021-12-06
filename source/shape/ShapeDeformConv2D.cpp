//
//  ShapeDeformConv2D.cpp
//  MNN
//
//  Created by MNN on 2021/10/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
class DeformConv2DSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(inputs.size() == 3);
        MNN_ASSERT(outputs.size() == 1);
        MNN_ASSERT(inputs[0]->length(0) == inputs[1]->length(0) && inputs[0]->length(0) == inputs[2]->length(0));
        MNN_ASSERT(inputs[1]->length(1) == 2 * inputs[2]->length(1));
        MNN_ASSERT(inputs[1]->length(2) == inputs[2]->length(2));
        MNN_ASSERT(inputs[1]->length(3) == inputs[2]->length(3));

        auto input = inputs[0];
        //auto offset = inputs[1];
        //auto mask = inputs[2];

        if (input->dimensions() <= 1) {
            // Deformable Convolution is not valid for dimension <= 1
            return false;
        }

        const auto common = op->main_as_DeformConv2D()->common();
        auto outputCount = common->outputCount();
        MNN_ASSERT(outputCount == outputs[0]->length(1));

        int kX = common->kernelX();
        int kY = common->kernelY();
        int kernel_width = common->dilateX() * (kX - 1) + 1;
        int kernel_height = common->dilateY() * (kY - 1) + 1;

        int output_width = 1;
        int output_height = 1;
        if (common->padMode() == PadMode_SAME) {
            // Tensorflow padding mode SAME
            output_width = ceil((float)input->width() / (float)common->strideX());
            output_height = ceil((float)input->height() / (float)common->strideY());
        }
        else if (common->padMode() == PadMode_VALID) {
            // Tensorflow padding mode VALID
            output_width = ceil((float)(input->width() - kernel_width + 1) / (float)common->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)common->strideY());
        }
        else {
            // Pad_Caffe means User setted padding
            if (nullptr != common->pads()) {
                MNN_ASSERT(common->pads()->size() >= 4);
                int input_width = input->width() + common->pads()->data()[1] + common->pads()->data()[3];
                int input_height = input->height() + common->pads()->data()[0] + common->pads()->data()[2];
                output_width = input_width < kernel_width ? 0 : (input_width - kernel_width) / common->strideX() + 1;
                output_height = input_height < kernel_height ? 0 : (input_height - kernel_height) / common->strideY() + 1;
            }
            else {
                int input_width = input->width() + common->padX() * 2;
                int input_height = input->height() + common->padY() * 2;
                output_width = (input_width - kernel_width) / common->strideX() + 1;
                output_height = (input_height - kernel_height) / common->strideY() + 1;
            }
        }

        auto &outputBuffer = outputs[0]->buffer();
        outputBuffer.dimensions = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        outputBuffer.type = input->getType();
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[3].extent = outputCount;
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
        }
        else {
            outputBuffer.dim[1].extent = outputCount;
            outputBuffer.dim[2].extent = output_height;
            outputBuffer.dim[3].extent = output_width;
        }
        // MNN_PRINT("outputs: %d, %d, %d, %d\n", outputs[0]->length(0), outputs[0]->length(1), outputs[0]->length(2), outputs[0]->length(3));
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        const auto common = op->main_as_DeformConv2D()->common();
        auto kw = common->kernelX();
        auto kh = common->kernelY();
        auto groups = common->groups();
        auto deform_groups = common->deform_groups();
        auto ic = inputs[0]->channel();
        auto oc = outputs[0]->channel();
        auto oSize = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
        if (common->inputCount() != ic && common->inputCount() > 0) {
            groups = ic / common->inputCount();
        }
        auto flops = (float)oSize * deform_groups * kw * kh * (ic * oc / (groups == 0 ? 1 : groups)) / FLOPS_M;
        return flops;
    }
};

REGISTER_SHAPE(DeformConv2DSizeComputer, OpType_DeformConv2D);

}// namespace MNN