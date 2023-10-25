/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tf_fast_rnnt/csrc/mutual_information.h"

#include <algorithm>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor_shape.h"

REGISTER_OP("FastRNNTLoss")
    .Input("px: float32")
    .Input("py: float32")
    .Input("boundary: int64")
    .Input("return_grad: bool")
    .Output("ans: float32")
    .Output("px_grad: float32")
    .Output("py_grad: float32");

REGISTER_OP("Cummin")
    .Input("in: int64")
    .Output("out: int64");

namespace tf = tensorflow;

namespace tf_fast_rnnt {
class FastRNNTOpBase : public tf::OpKernel {
  public:
    explicit FastRNNTOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    }

    void Compute(tf::OpKernelContext* ctx) override {
        // Grab the input tensors
        const tf::Tensor* px;
        const tf::Tensor* py;
        const tf::Tensor* boundary;
        const tf::Tensor* return_grad;
        OP_REQUIRES_OK(ctx, ctx->input("px", &px));
        OP_REQUIRES_OK(ctx, ctx->input("py", &py));
        OP_REQUIRES_OK(ctx, ctx->input("boundary", &boundary));
        OP_REQUIRES_OK(ctx, ctx->input("return_grad", &return_grad));
        auto stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        const int B = px->dim_size(0), S = px->dim_size(1), T = py->dim_size(2);

        auto px_t = px->tensor<float, 3>();
        auto py_t = py->tensor<float, 3>();
        auto boundary_t = boundary->matrix<int64_t>();
        auto return_grad_t = return_grad->scalar<bool>();

        tf::Tensor p;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, tf::TensorShape({B, S+1, T+1}), &p));
        auto p_t = p.tensor<float, 3>();

        tf::Tensor* ans = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("ans", tf::TensorShape({B}), &ans));
        auto ans_t = ans->vec<float>();

        // compute RNNT
        int status = tf_fast_rnnt::MutualInformationCuda<float>(
                              px_t,
                              py_t,
                              boundary_t,
                              p_t,
                              ans_t,
                              stream);

        tf::Tensor* px_grad = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("px_grad", tf::TensorShape({B, S, T+1}), &px_grad));

        tf::Tensor* py_grad = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("py_grad", tf::TensorShape({B, S+1, T}), &py_grad));

        if(return_grad_t()) {      
          tf::Tensor p_grad;
          OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, tf::TensorShape({B, S+1, T+1}), &p_grad));
          auto p_grad_t = p_grad.tensor<float, 3>();

          set_zero(px_grad, stream);
          auto px_grad_t = px_grad->tensor<float, 3>();

          set_zero(py_grad, stream);
          auto py_grad_t = py_grad->tensor<float, 3>();

          tf::Tensor ans_grad;
          OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, tf::TensorShape({B}), &ans_grad));
          auto ans_grad_t = ans_grad.vec<float>();

          // set all values of ans_grad to 1
          float tmp[B];
          std::fill_n(tmp, B, 1.);
          cudaMemcpyAsync(ans_grad_t.data(), tmp, sizeof(float) * B, cudaMemcpyHostToDevice, stream);

          tf_fast_rnnt::MutualInformationBackwardCuda<float>(px_t, py_t, boundary_t, p_t,
                                        p_grad_t, px_grad_t, py_grad_t, ans_grad_t, true, stream);

        }
        cudaStreamSynchronize(stream);                              
        OP_REQUIRES(ctx, status == 1,
                    tf::errors::Internal("rnnt_loss error in compute_rnnt_loss: "
                                         ));
    }
  private:
    virtual void set_zero(tf::Tensor* t, cudaStream_t stream) = 0;
};

class FastRNNTOpGPU : public FastRNNTOpBase {
  public:
    explicit FastRNNTOpGPU(tf::OpKernelConstruction* ctx) : FastRNNTOpBase(ctx) {
    }
  private:
    void set_zero(tf::Tensor* t, cudaStream_t stream) override {
        cudaMemsetAsync(t->flat<float>().data(), 0, t->NumElements()*sizeof(float), stream);
    }
};
REGISTER_KERNEL_BUILDER(Name("FastRNNTLoss").Device(::tensorflow::DEVICE_GPU)
                        .HostMemory("return_grad"),
                        FastRNNTOpGPU);

class CumminOpGPU : public tf::OpKernel {
  public:
    explicit CumminOpGPU(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
    }

    void Compute(tf::OpKernelContext* ctx) override {
        // Grab the input tensors
        const tf::Tensor* in;
        OP_REQUIRES_OK(ctx, ctx->input("in", &in));

        auto stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        const int B = in->dim_size(0), S = in->dim_size(1);

        auto in_t = in->tensor<int64_t, 2>();

        tf::Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("out", tf::TensorShape({B, S}), &out));
        auto out_t = out->tensor<int64_t, 2>();

        int status = tf_fast_rnnt::CumminCuda<int64_t>(in_t,
                              out_t,
                              stream);
        cudaStreamSynchronize(stream);                              
        OP_REQUIRES(ctx, status == 1,
                    tf::errors::Internal("rnnt_loss error in cummin: "
                                         ));
    }
};
REGISTER_KERNEL_BUILDER(Name("Cummin").Device(::tensorflow::DEVICE_GPU),
                        CumminOpGPU);
}

#undef EIGEN_USE_GPU
