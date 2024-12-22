using namespace torch::indexing;
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define NUM_THREADS 256

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void conv_kernel(const c10::complex<float>* fdl, const c10::complex<float>* filters_fd, const int fdl_cursor, c10::complex<float>* output_fd, const int K, const int B, const int C) {
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id >= C * (B + 1)) return;

    const int channel_id = thread_id / (B + 1);
    const int bin_id = thread_id % (B + 1);
    int cursor = fdl_cursor;

    const int fdl_offset = bin_id * K;
    const int filter_offset = channel_id * ((B + 1) * K) + bin_id * K;
    const int output_offset = channel_id * (B + 1) + bin_id;

    c10::complex<float> out = 0;
    for (int k = 0; k < K; ++k) {
        out += fdl[fdl_offset + cursor] * filters_fd[filter_offset + k];
        cursor = (cursor - 1 + K) % K;
    }
    output_fd[output_offset] = out;
}

torch::Tensor part_conv_gpu(torch::Tensor input_fd, torch::Tensor fdl, torch::Tensor filters_fd, const int fdl_cursor, const int K, const int B, const int C) {
    CHECK_INPUT(input_fd);
    CHECK_INPUT(fdl);
    CHECK_INPUT(filters_fd);

    auto output_fd = torch::empty({C, B+1}, input_fd.options());

    const int blocks = cdiv(C * (B + 1), NUM_THREADS);

    // Store the fd signal in a frequency-domain delay line
    fdl.index_put_({Slice(0, B+1), fdl_cursor}, input_fd);

    conv_kernel<<<blocks, NUM_THREADS>>>(fdl.data_ptr<c10::complex<float>>(), filters_fd.data_ptr<c10::complex<float>>(), fdl_cursor, output_fd.data_ptr<c10::complex<float>>(), K, B, C);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_fd;
}