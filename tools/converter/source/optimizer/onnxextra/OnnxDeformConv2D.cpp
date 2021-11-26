#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxDeformConv2DTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        const int inputSize = inputs.size();
        if (inputSize != 5) {
            MNN_ERROR("Deformable Convolution Input Error!\n");
        }

        // inputs: [input, offset, mask, weight, bias]
        auto weight = inputs[3];
        auto weightInfo = weight->getInfo();
        if (nullptr == weightInfo) {
            MNN_ERROR("Deformable Convolution should know weight shape infromation!\n");
            return nullptr;
        }
        auto &weightShape = weightInfo->dim;

        auto op = expr->get();
        auto extraParam = op->main_as_Extra();
        std::string originalOpType(extraParam->type()->c_str());

        int co = weightShape[0];
        int ci = weightShape[1];
        int kh = weightShape[2];
        int kw = 1;
        if (weightShape.size() >= 4) {
            kw = weightShape[3];
        }

        int groups = 1;
        int deform_groups = 1;
        int dilation_h = 1;
        int dilation_w = 1;
        int stride_h = 1;
        int stride_w = 1;
        int padding_x = 1;
        int padding_y = 1;
        std::vector<int> inputPadding(4);

        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr = extraParam->attr()->GetAs<Attribute>(i);
            const auto &key = attr->key()->str();
            if (key == "dilation") {
                auto dataList = attr->list();
                dilation_h = dataList->i()->data()[0];
                if (dataList->i()->size() >= 2) {
                    dilation_w = dataList->i()->data()[1];
                }
            }
            else if (key == "stride") {
                auto dataList = attr->list();
                stride_h = dataList->i()->data()[0];
                if (dataList->i()->size() >= 2) {
                    stride_w = dataList->i()->data()[1];
                }
            }
            else if (key == "padding") {
                auto dataList = attr->list();
                padding_y = dataList->i()->data()[0];
                if (dataList->i()->size() >= 2) {
                    padding_x = dataList->i()->data()[1];
                }
            }
            else if (key == "groups") {
                groups = attr->i();
            }
            else if (key == "deform_groups") {
                deform_groups = attr->i();
            }
        }

        inputPadding[0] = padding_y;
        inputPadding[1] = padding_x;
        inputPadding[2] = padding_y;
        inputPadding[3] = padding_x;

        std::unique_ptr<DeformConv2DT> deformConv2dParam(new MNN::DeformConv2DT);
        deformConv2dParam->common.reset(new MNN::DeformConv2DCommonT);
        auto common = deformConv2dParam->common.get();

        // set parameters
        common->padX = padding_x;
        common->padY = padding_y;
        common->kernelX = kw;
        common->kernelY = kh;
        common->strideX = stride_w;
        common->strideY = stride_h;
        common->dilateX = dilation_w;
        common->dilateY = dilation_h;
        common->padMode = PadMode_CAFFE;
        common->groups = groups;
        common->deform_groups = deform_groups;
        common->outputCount = co;
        common->inputCount= ci * groups;
        common->relu = false;
        common->relu6 = false;
        common->pads = inputPadding;

        auto config = Global<modelConfig>::Get();
        // read weight data
        const float *weightDataPtr = nullptr;
        int limitNumber = 4;
        if (config->optimizePrefer == 1) {
            // Smallest
            limitNumber = 1;
        }
        else if (config->optimizePrefer == 2) {
            // Fastest
            limitNumber = 100;
        }
        if (weight->linkNumber() <= limitNumber) {
            weightDataPtr = weight->readMap<float>();
        }

        // weight is Constant node
        if (weightDataPtr) {
            if (weight->linkNumber() > 1) {
                static bool gPrint = false;
                if (!gPrint) {
                    MNN_PRINT("The Deformable Convolution use shared weight, may increase the model size\n");
                    gPrint = true;
                }
            }

            // MNN_PRINT("MNNCountNNZBlock:%p\n", MNNCountNNZBlock);
            const size_t weightSize = co * ci * kh * kw;
            deformConv2dParam->weight.resize(weightSize);
            ::memcpy(deformConv2dParam->weight.data(), weightDataPtr, weightSize * sizeof(float));
            deformConv2dParam->bias.resize(common->outputCount);
            if (inputSize == 5) {
                // read bias data
                auto bias = inputs[4];
                const int biasNums = bias->getInfo()->size;
                if (biasNums != common->outputCount) {
                    // TODO broacast
                    MNN_ERROR("[TODO] Deformable Conv's bias support broadcast!\n");
                    return nullptr;
                }
                auto biasDataPtr = bias->readMap<float>();
                if (!biasDataPtr) {
                    MNN_ERROR("Deformable Conv's bias input should be Constant!\n");
                    return nullptr;
                }
                ::memcpy(deformConv2dParam->bias.data(), biasDataPtr, common->outputCount * sizeof(float));
            }
            else {
                ::memset(deformConv2dParam->bias.data(), 0, common->outputCount * sizeof(float));
            }
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name = expr->name();
        newOp->type = OpType_DeformConv2D;
        
        newOp->main.type = OpParameter_DeformConv2D;
        newOp->main.value = deformConv2dParam.release();
        auto x = inputs[0];
        EXPRP deformConv2DExpr;
        if (weightDataPtr) {
            // merge weight(bias) node to DeformConv2D parameter
            deformConv2DExpr = Expr::create(newOp.get(), { x , inputs[1], inputs[2] });
        }
        else {
            // include weight bias 
            if (inputs.size() > 4) {
                deformConv2DExpr = Expr::create(newOp.get(), { x , inputs[1], inputs[2], inputs[3], inputs[4] });
            }
            else {
                deformConv2DExpr = Expr::create(newOp.get(), { x , inputs[1], inputs[2], inputs[3] });
            }
        }

        deformConv2DExpr->setName(expr->name());
        auto res = Variable::create(deformConv2DExpr);

        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("DeformConv2D", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxDeformConv2DTransform));
    return true;
}();

}// namespace Express
}// namespace MNN