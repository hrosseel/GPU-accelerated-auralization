using namespace torch::indexing;
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void conv_kernel(const c10::complex<double>* fdl, const c10::complex<double>* filters_fd, int fdl_cursor, c10::complex<double>* output_fd, int K, int B, int C) {
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id >= C * (B + 1)) return;

    const int channel_id = thread_id / (B + 1);
    const int bin_id = thread_id % (B + 1);
    int cursor = fdl_cursor;

    const int fdl_offset = bin_id * K;
    const int filter_offset = channel_id * ((B + 1) * K) + bin_id * K;
    const int output_offset = channel_id * (B + 1) + bin_id;

    c10::complex<double> out = 0;
    for (int k = 0; k < K; ++k) {
        out += fdl[fdl_offset + cursor] * filters_fd[filter_offset + k];
        cursor = (cursor - 1 + K) % K;
    }
    output_fd[output_offset] = out;
}

torch::Tensor part_conv_gpu(torch::Tensor input_fd, torch::Tensor fdl, torch::Tensor filters_fd, int fdl_cursor, int K, int B, int C) {
    CHECK_INPUT(input_fd);
    CHECK_INPUT(fdl);
    CHECK_INPUT(filters_fd);

    auto output_fd = torch::empty({C, B+1}, input_fd.options());

    int threads = 256;
    int blocks = cdiv(C * (B + 1), threads);

    // Store the fd signal in a frequency-domain delay line
    fdl.index_put_({Slice(0, B+1), fdl_cursor}, input_fd);

    conv_kernel<<<blocks, threads>>>(fdl.data_ptr<c10::complex<double>>(), filters_fd.data_ptr<c10::complex<double>>(), fdl_cursor, output_fd.data_ptr<c10::complex<double>>(), K, B, C);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_fd;
}