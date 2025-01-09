using namespace torch::indexing;
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#ifndef NUM_CHANNELS
    #error "NUM_CHANNELS is not defined"
#endif
#ifndef BLOCK_SIZE
    #error "BLOCK_SIZE is not defined"
#endif
#ifndef NUM_PARTS
    #error "NUM_PARTS is not defined"
#endif
#ifndef NUM_THREADS
    #define NUM_THREADS 256
#endif

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void conv_kernel(const c10::complex<float>* fdl, const c10::complex<float>* filters_fd, int fdl_cursor, c10::complex<float>* output_fd) {
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id >= NUM_CHANNELS * (BLOCK_SIZE + 1)) return;

    const int channel_id = thread_id / (BLOCK_SIZE + 1);
    const int bin_id = thread_id % (BLOCK_SIZE + 1);
    int cursor = fdl_cursor;

    #ifdef MULTI_INPUT  // Multi-input mode  
    const int fdl_offset = channel_id * ((BLOCK_SIZE + 1) * NUM_PARTS) + bin_id * NUM_PARTS;
    const int filter_offset = fdl_offset;
    #else  // Single-input mode
    const int fdl_offset = bin_id * NUM_PARTS;
    const int filter_offset = channel_id * ((BLOCK_SIZE + 1) * NUM_PARTS) + bin_id * NUM_PARTS;
    #endif

    const int output_offset = channel_id * (BLOCK_SIZE + 1) + bin_id;
    c10::complex<float> out = 0;

    for (int k = 0; k < NUM_PARTS; ++k) {
        out += fdl[fdl_offset + cursor] * filters_fd[filter_offset + k];
        cursor = (cursor - 1 + NUM_PARTS) % NUM_PARTS;
    }
    output_fd[output_offset] = out;
}


torch::Tensor part_conv_gpu(torch::Tensor input_fd, torch::Tensor fdl, torch::Tensor filters_fd, int fdl_cursor) {
    CHECK_INPUT(input_fd);
    CHECK_INPUT(fdl);
    CHECK_INPUT(filters_fd);

    auto output_fd = torch::empty({NUM_CHANNELS, BLOCK_SIZE+1}, input_fd.options());

    int blocks = cdiv(NUM_CHANNELS * (BLOCK_SIZE + 1), NUM_THREADS);

    // Store the fd signal in a frequency-domain delay line
    fdl.index_put_({Slice(), Slice(0, BLOCK_SIZE + 1), fdl_cursor}, input_fd);

    // Perform the convolution
    conv_kernel<<<blocks, NUM_THREADS>>>(fdl.data_ptr<c10::complex<float>>(), filters_fd.data_ptr<c10::complex<float>>(), fdl_cursor, output_fd.data_ptr<c10::complex<float>>());

    C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for errors
    return output_fd;
}