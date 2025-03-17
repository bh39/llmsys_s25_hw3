#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1: 
  float l_sum = 0;
  float l2_sum = 0;

  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l2_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2: 
  blockReduce<ReduceType::kSum, 1>(&l_sum);
  __syncthreads();
  blockReduce<ReduceType::kSum, 1>(&l2_sum);
  
  __shared__ float s_mean, s_variance;
  if (threadIdx.x == 0) {
    s_mean = l_sum / (hidden_size * 4);
    s_variance = l2_sum / (hidden_size * 4) - s_mean * s_mean + LN_EPSILON;
    
    vars[blockIdx.x] = s_variance;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
  }
  __syncthreads();
  
  // inverse std
  float inv_std = rsqrtf(s_variance);
  
  // Step 3:
  float4 *output_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 scale_val = scale_f4[idx];
    float4 bias_val = bias_f4[idx];
    
    float4 result;
    // normalization: 
    result.x = scale_val.x * ((val.x - s_mean) * inv_std) + bias_val.x;
    result.y = scale_val.y * ((val.y - s_mean) * inv_std) + bias_val.y;
    result.z = scale_val.z * ((val.z - s_mean) * inv_std) + bias_val.z;
    result.w = scale_val.w * ((val.w - s_mean) * inv_std) + bias_val.w;
    
    output_f4[idx] = result;
  }
  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct thread_idxition in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
  // Step 1
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  float dbetta = 0.0f;
  float dgamma = 0.0f;

  if (thread_idx < width) {
    for (int i = threadIdx.y; i < rows; i += TILE_DIM) {
      int idx = i * width + thread_idx;
      float dout = (float)out_grad[idx];
      float xhat;

      if (means != nullptr) {
        // get xhat from means
        xhat = ((float)inp[idx] - means[i]) * rsqrtf(vars[i]);
      } else {
        // get xhat from beta and gamma
        xhat = ((float)inp[idx] - betta[thread_idx]) / gamma[thread_idx];
      }

      dbetta += dout;
      dgamma += xhat * dout;
    }
  }
  
  // Step 2: 
  betta_buffer[threadIdx.x][threadIdx.y] = dbetta;
  gamma_buffer[threadIdx.x][threadIdx.y] = dgamma;
  
  __syncthreads();

  // Step 3: transthread_idxe to get values for the same column together for shuffle
  float betta_val = betta_buffer[threadIdx.y][threadIdx.x];
  float gamma_val = gamma_buffer[threadIdx.y][threadIdx.x];
  
  // shfl_down to reduce
  for (int i = 1; i < TILE_DIM; i *= 2) {
    betta_val += g.shfl_down(betta_val, i);
    gamma_val += g.shfl_down(gamma_val, i);
  }
  // Step 4:
  if (threadIdx.x == 0 && thread_idx < width) {
    betta_grad[blockIdx.x * blockDim.x + threadIdx.y] = betta_val;
    gamma_grad[blockIdx.x * blockDim.x + threadIdx.y] = gamma_val;
  }
  
  /// END ASSIGN3_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  // Step 1: 
  int thread_idx = blockIdx.x * hidden_dim + threadIdx.x;
  if (threadIdx.x >= hidden_dim) {
    return;
  }
  
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad);
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp);
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad);
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *betta_f4 = (betta != nullptr) ? reinterpret_cast<const float4 *>(betta) : nullptr;
  
  float4 dxhat = out_grad_f4[thread_idx];
  float4 gamma_val = gamma_f4[threadIdx.x];

  dxhat.x *= gamma_val.x;
  dxhat.y *= gamma_val.y;
  dxhat.z *= gamma_val.z;
  dxhat.w *= gamma_val.w;
  
  // Step 2: 
  float4 xhat = inp_f4[thread_idx];
  float inv_std = rsqrtf(vars[blockIdx.x]);
  
  if (means != nullptr) {
    // xhat from means
    float mean_val = means[blockIdx.x];
    xhat.x = (xhat.x - mean_val) * inv_std;
    xhat.y = (xhat.y - mean_val) * inv_std;
    xhat.z = (xhat.z - mean_val) * inv_std;
    xhat.w = (xhat.w - mean_val) * inv_std;
  } else {
    // xhat from betta and gamma
    float4 betta_val = betta_f4[threadIdx.x];
    xhat.x = (xhat.x - betta_val.x) / gamma_val.x;
    xhat.y = (xhat.y - betta_val.y) / gamma_val.y;
    xhat.z = (xhat.z - betta_val.z) / gamma_val.z;
    xhat.w = (xhat.w - betta_val.w) / gamma_val.w;
  }
  
  // Step 3: 
  float l_dxhat_sum, l_dxhat_xhat_sum;
  l_dxhat_sum = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
  l_dxhat_xhat_sum = dxhat.x * xhat.x + dxhat.y * xhat.y + 
                 dxhat.z * xhat.z + dxhat.w * xhat.w;
  
  // sync blockReduce according to edstem
  blockReduce<ReduceType::kSum, 1>(&l_dxhat_sum);
  __syncthreads();
  blockReduce<ReduceType::kSum, 1>(&l_dxhat_xhat_sum);
  __syncthreads();
  
  __shared__ float s_dxhat_sum, s_dxhat_xhat_sum;
  if (threadIdx.x == 0) {
    float scale = 1.0f / (hidden_dim * 4);
    s_dxhat_sum = l_dxhat_sum * scale;
    s_dxhat_xhat_sum = l_dxhat_xhat_sum * scale;
  }
  __syncthreads();
  
  // Step 4: 
  dxhat.x = (dxhat.x - s_dxhat_sum - xhat.x * s_dxhat_xhat_sum) * inv_std;
  dxhat.y = (dxhat.y - s_dxhat_sum - xhat.y * s_dxhat_xhat_sum) * inv_std;
  dxhat.z = (dxhat.z - s_dxhat_sum - xhat.z * s_dxhat_xhat_sum) * inv_std;
  dxhat.w = (dxhat.w - s_dxhat_sum - xhat.w * s_dxhat_xhat_sum) * inv_std;
  
  inp_grad_f4[thread_idx] = dxhat;

  /// END ASSIGN3_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
