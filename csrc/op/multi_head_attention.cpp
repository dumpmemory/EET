#include "op/multi_head_attention.hpp"
#include "core/add_bias.cuh"
#include "core/transpose.cuh"
#include "core/layer_norm.cuh"
#include "core/self_add_bias.cuh"
#include "core/attention_dispatch.cuh"
#include "core/bert_softmax.cuh"
#include "core/pre_process.cuh"

// for gpt
namespace eet{
    namespace op{
        MultiHeadAttention::MultiHeadAttention(MetaDesc desc,
                                    const torch::Tensor& Q_weights,
                                    const torch::Tensor& K_weights,
                                    const torch::Tensor& V_weights,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias):
            desc_(desc),
            max_len_(0),
            q_weights_(Q_weights.data_ptr()),
            k_weights_(K_weights.data_ptr()),
            v_weights_(V_weights.data_ptr()),
            q_bias_(Q_bias.data_ptr()),
            k_bias_(K_bias.data_ptr()),
            v_bias_(V_bias.data_ptr()),
            output_weights_(Output_weights.data_ptr()),
            output_bias_(Output_bias.data_ptr()),
            layernorm_weights_(layernorm_weights.data_ptr()),
            layernorm_bias_(layernorm_bias.data_ptr())
        {   
            size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
            output_ = torch::zeros({desc_.batch_size_, desc_.max_seq_len_, desc_.hidden_units_}, desc_.options_);
            check_cuda_error(cudaMalloc(&fused_qkv_ptr_,sizeof(void**) * FUSED_QKV_PTR_SIZE));
            qkv_kernel_ = (void**)fused_qkv_ptr_;
            qkv_input_  = qkv_kernel_ + QKV_PTR_SIZE;
            qkv_buf_   = qkv_input_  + QKV_PTR_SIZE;

            switch (desc_.dtype_)
            {
            case torch::kFloat32:
                qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT;
                q_k_algo_ = CUBLAS_GEMM_DEFAULT;
                attn_v_algo_ = CUBLAS_GEMM_DEFAULT;
                alpha_ = new float();
                beta_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                break;
            case torch::kFloat16:
                qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                q_k_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                attn_v_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                alpha_ = new half();
                beta_ = new half();
                *((half *)alpha_) = (half)1.0f;
                *((half *)beta_) = (half)0.0f;
                break;
            //TODO
            case torch::kInt8:
                break;
            }
        }

        // encoder
        torch::Tensor MultiHeadAttention::forward(torch::Tensor& input,
                                    const torch::Tensor& padding_mask,
                                    bool pre_layernorm,
                                    bool add_redusial){
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as MultiHeadAttention's dtype");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];

            Buffer& q_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& k_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& v_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_);
                layer_norm(input,layernormed_query);

                //qkv * weights
                qkv_weights_mul(layernormed_query.data_ptr(), q_buffer,k_buffer,v_buffer);
                layernormed_query.free();
            }
            else{
                //qkv * weights
                qkv_weights_mul(input.data_ptr(), q_buffer,k_buffer,v_buffer);
            }

            //qkv add bias                
            Buffer& q_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& k_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& v_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            qkv_add_bias(q_buffer, k_buffer, v_buffer, q_buf, k_buf, v_buf);
            
            q_buffer.free();
            k_buffer.free();
            v_buffer.free();

            //q * k
            Buffer& qk_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.head_num_ *
                                        desc_.max_full_seq_len_ * desc_.max_full_seq_len_, desc_.dtype_, desc_.options_);
            q_k_mul(q_buf, k_buf, qk_buf);

            q_buf.free();

            //softmax
            if (padding_mask.data_ptr() == nullptr)
            {
                Buffer& atten_mask = MManager::get_instance().get_buffer(desc_.batch_size_* desc_.max_full_seq_len_ * desc_.max_full_seq_len_, torch::kInt64, desc_.options_);
                fill_kernel((int64_t*)atten_mask.data_ptr(),cur_batch_size_* cur_seq_len_* cur_seq_len_,(int64_t)1);
                qk_softmax(qk_buf,atten_mask.data_ptr());
                atten_mask.free();
            }
            else{
                qk_softmax(qk_buf,padding_mask.data_ptr());
            }



            //attn * v
            Buffer& transpose_dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            
            attn_v_mul(qk_buf,v_buf,transpose_dst);

            qk_buf.free();
            k_buf.free();
            v_buf.free();

            //transpose
            Buffer& dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);

            transpose(transpose_dst, dst);
            transpose_dst.free();

            //project
            project(dst,output_,input ,pre_layernorm,add_redusial);
            dst.free();
            // output_ = output_[input.sizes()];

            auto res = torch::from_blob(output_.data_ptr(), input.sizes(), input.strides(), desc_.options_);

            return std::move(res);
        }

        // layerNorm
        void MultiHeadAttention::layer_norm(const torch::Tensor& input, Buffer& layernorm_query)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;

            RUN_KERNEL(layernorm,desc_.dtype_,input.data_ptr(),layernorm_weights_,layernorm_bias_,layernorm_query.data_ptr(), m, n, desc_.stream);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::qkv_weights_mul(void* input, 
                                    Buffer& q_buffer,
                                    Buffer& k_buffer,
                                    Buffer& v_buffer){

                const int m = cur_batch_size_ * cur_seq_len_;
                const int k = desc_.hidden_units_;
                const int n = k;
                const void *hA[]{q_weights_,k_weights_,v_weights_,
                                input, input, input,
                                q_buffer.data_ptr(), k_buffer.data_ptr(), v_buffer.data_ptr()};
                check_cuda_error(cudaMemcpyAsync((void *)qkv_kernel_, hA, sizeof(void *) * FUSED_QKV_PTR_SIZE, cudaMemcpyHostToDevice));

                check_cuda_error(cublasGemmBatchedEx(desc_.cublasHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            alpha_,
                            (const void *const *)qkv_kernel_, desc_.computeType_, n,
                            (const void *const *)qkv_input_, desc_.computeType_, k,
                            beta_,
                            (void *const *)qkv_buf_, desc_.computeType_, n,
                            QKV_PTR_SIZE,
                            desc_.computeType_,
                            q_k_algo_));
#ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::qkv_add_bias(const Buffer &q_buffer,
                                                       const Buffer &k_buffer,
                                                       const Buffer &v_buffer,
                                                       Buffer &q_buf,
                                                       Buffer &k_buf,
                                                       Buffer &v_buf)
        {

            RUN_KERNEL(add_QKV_bias_opt_kernel, desc_.dtype_, q_buffer.data_ptr(),q_bias_,
                        k_buffer.data_ptr(), k_bias_, v_buffer.data_ptr(), v_bias_,
                        q_buf.data_ptr(), k_buf.data_ptr(), v_buf.data_ptr(),
                        cur_batch_size_, cur_seq_len_, desc_.head_num_, size_per_head_, desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::q_k_mul(const Buffer& q_buf, const Buffer& k_buf,
                                                Buffer& qk_buf){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                cur_seq_len_, cur_seq_len_, size_per_head_,
                alpha_,
                k_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                q_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                beta_,
                qk_buf.data_ptr(), desc_.computeType_, cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                cur_batch_size_ * desc_.head_num_,
                desc_.computeType_,
                q_k_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::qk_softmax(Buffer& qk_buf,void* attr_mask){
            float scalar = 1 / sqrtf(size_per_head_ * 1.0f);
            RUN_KERNEL(bert_softmax_kernel,desc_.dtype_,qk_buf.data_ptr(), attr_mask, cur_batch_size_,
                    desc_.head_num_,cur_seq_len_, scalar, desc_.stream);
#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 
        }
        

        void MultiHeadAttention::attn_v_mul(const Buffer& qk_buf,
                                             const Buffer& v_buf,
                                             Buffer& transpose_dst){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size_per_head_, cur_seq_len_, cur_seq_len_,
                    alpha_,
                    v_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                    qk_buf.data_ptr(), desc_.computeType_, cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                    beta_,
                    transpose_dst.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                    cur_batch_size_ * desc_.head_num_,
                    desc_.computeType_,
                    attn_v_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::transpose(const Buffer& transpose_dst, Buffer&  dst){
            RUN_KERNEL(transpose_kernel,desc_.dtype_,transpose_dst.data_ptr(),dst.data_ptr(), cur_batch_size_, cur_seq_len_,
                    desc_.head_num_, size_per_head_, desc_.stream);

            #ifdef _DEBUG_MODE_
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
            #endif
        }

        void MultiHeadAttention::project(const Buffer& dst, torch::Tensor& res,torch::Tensor& input, bool pre_layernorm,bool add_redusial){
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.head_num_ * size_per_head_;
            int n = k;
            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            n, m, k,
                                            alpha_,
                                            output_weights_, desc_.computeType_, n,
                                            dst.data_ptr(), desc_.computeType_, k,
                                            beta_,
                                            res.data_ptr(), desc_.computeType_, n,
                                            desc_.computeType_,
                                            qkv_weights_algo_));
            if(add_redusial)
            {   
                if(!pre_layernorm)
                {   
                    // add_bias + add_redusial + layer_norm
                    RUN_KERNEL(add_bias_input_layernorm_kernel,desc_.dtype_,
                                        res.data_ptr(),input.data_ptr(), 
                                        output_bias_,layernorm_weights_,
                                        layernorm_bias_,m , n, desc_.stream);
                }
                else
                {
                    // add_bias + add_residual
                    RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, res.data_ptr(), input.data_ptr(),output_bias_,
                           m , n, desc_.stream);
                }
            }
            else
            {
                // only add bias
                RUN_KERNEL(add_bias_kernel, desc_.dtype_, res.data_ptr(), output_bias_,
                           m , n, desc_.stream);
            }
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }
    }
}