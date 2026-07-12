#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int get_packed_code_u8(
    const unsigned char* row,
    int64_t idx,
    int bits) {
    const int64_t bit_offset = idx * bits;
    const int64_t byte_idx = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7);
    unsigned int value = static_cast<unsigned int>(row[byte_idx]) >> shift;
    if (shift + bits > 8) {
        value |= static_cast<unsigned int>(row[byte_idx + 1]) << (8 - shift);
    }
    return static_cast<int>(value & ((1u << bits) - 1u));
}

__device__ __forceinline__ void set_packed_code_u8(
    unsigned char* row,
    int64_t idx,
    int bits,
    int code) {
    const int64_t bit_offset = idx * bits;
    const int64_t byte_idx = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7);
    const unsigned int mask = (1u << bits) - 1u;
    unsigned int value = (static_cast<unsigned int>(code) & mask) << shift;
    row[byte_idx] |= static_cast<unsigned char>(value & 0xffu);
    if (shift + bits > 8) {
        row[byte_idx + 1] |= static_cast<unsigned char>((value >> 8) & 0xffu);
    }
}

__global__ void pack_codes_kernel(
    const int64_t* __restrict__ codes,
    unsigned char* __restrict__ packed,
    int64_t rows,
    int64_t length,
    int64_t packed_cols,
    int bits) {
    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned char* out = packed + row * packed_cols;
    const int64_t* in = codes + row * length;
    for (int64_t j = 0; j < length; ++j) {
        set_packed_code_u8(out, j, bits, static_cast<int>(in[j]));
    }
}

__global__ void unpack_codes_kernel(
    const unsigned char* __restrict__ packed,
    int64_t* __restrict__ codes,
    int64_t rows,
    int64_t length,
    int64_t packed_cols,
    int bits) {
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * length;
    if (linear >= total) {
        return;
    }
    const int64_t row = linear / length;
    const int64_t col = linear - row * length;
    const unsigned char* in = packed + row * packed_cols;
    codes[linear] = static_cast<int64_t>(get_packed_code_u8(in, col, bits));
}

__global__ void unpack_codes_u8_kernel(
    const unsigned char* __restrict__ packed,
    unsigned char* __restrict__ codes,
    int64_t rows,
    int64_t length,
    int64_t packed_cols,
    int bits) {
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * length;
    if (linear >= total) {
        return;
    }
    const int64_t row = linear / length;
    const int64_t col = linear - row * length;
    const unsigned char* in = packed + row * packed_cols;
    codes[linear] = static_cast<unsigned char>(get_packed_code_u8(in, col, bits));
}

}  // namespace

torch::Tensor pack_codes_cuda(torch::Tensor codes, int64_t bits) {
    TORCH_CHECK(codes.is_cuda(), "codes must be CUDA tensor");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(codes.dim() == 2, "codes must have shape [rows, length]");
    auto codes_i64 = codes.to(torch::kInt64).contiguous();
    const int64_t rows = codes_i64.size(0);
    const int64_t length = codes_i64.size(1);
    const int64_t packed_cols = (length * bits + 7) / 8;
    auto packed = torch::zeros({rows, packed_cols}, codes.options().dtype(torch::kUInt8));
    const int threads = 128;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    pack_codes_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        codes_i64.data_ptr<int64_t>(),
        packed.data_ptr<unsigned char>(),
        rows,
        length,
        packed_cols,
        static_cast<int>(bits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor unpack_codes_cuda(torch::Tensor packed, int64_t bits, int64_t length) {
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(packed.dim() == 2, "packed must have shape [rows, packed_cols]");
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    auto codes = torch::empty({rows, length}, packed.options().dtype(torch::kInt64));
    const int threads = 256;
    const int64_t total = rows * length;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    unpack_codes_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<unsigned char>(),
        codes.data_ptr<int64_t>(),
        rows,
        length,
        packed_cols,
        static_cast<int>(bits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return codes;
}

torch::Tensor unpack_codes_u8_cuda(torch::Tensor packed, int64_t bits, int64_t length) {
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(packed.dim() == 2, "packed must have shape [rows, packed_cols]");
    const int64_t rows = packed.size(0);
    const int64_t packed_cols = packed.size(1);
    auto codes = torch::empty({rows, length}, packed.options().dtype(torch::kUInt8));
    const int threads = 256;
    const int64_t total = rows * length;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    unpack_codes_u8_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<unsigned char>(),
        codes.data_ptr<unsigned char>(),
        rows,
        length,
        packed_cols,
        static_cast<int>(bits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return codes;
}
