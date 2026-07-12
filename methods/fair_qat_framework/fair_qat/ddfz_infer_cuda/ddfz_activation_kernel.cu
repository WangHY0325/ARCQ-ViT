#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

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

__global__ void quantize_activation_kernel(
    const float* __restrict__ x,
    const float* __restrict__ thresholds,
    const float* __restrict__ codebook,
    unsigned char* __restrict__ packed,
    float* __restrict__ center,
    float* __restrict__ scale,
    float* __restrict__ code_sum,
    int64_t rows,
    int64_t cols,
    int64_t groups,
    int64_t packed_cols,
    int bits,
    int group_size,
    int levels) {
    const int64_t linear_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_groups = rows * groups;
    if (linear_group >= total_groups) {
        return;
    }

    const int64_t row = linear_group / groups;
    const int64_t group = linear_group - row * groups;
    const int64_t offset = row * cols + group * group_size;

    float mu = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        mu += x[offset + j];
    }
    mu /= static_cast<float>(group_size);

    float var = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        const float r = x[offset + j] - mu;
        var += r * r;
    }
    const float s = sqrtf(var / static_cast<float>(group_size) + 1.0e-6f);

    float sum_codes = 0.0f;
    unsigned char* out_row = packed + row * packed_cols;
    for (int j = 0; j < group_size; ++j) {
        const float t = (x[offset + j] - mu) / s;
        int code = 0;
        while (code < levels - 1 && t > thresholds[code]) {
            ++code;
        }
        set_packed_code_u8(out_row, group * group_size + j, bits, code);
        sum_codes += codebook[code];
    }

    center[linear_group] = mu;
    scale[linear_group] = s;
    code_sum[linear_group] = sum_codes;
}

__global__ void quantize_activation_u8_kernel(
    const float* __restrict__ x,
    const float* __restrict__ thresholds,
    const float* __restrict__ codebook,
    unsigned char* __restrict__ codes,
    float* __restrict__ center,
    float* __restrict__ scale,
    float* __restrict__ code_sum,
    int64_t rows,
    int64_t cols,
    int64_t groups,
    int group_size,
    int levels) {
    const int64_t linear_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_groups = rows * groups;
    if (linear_group >= total_groups) {
        return;
    }

    const int64_t row = linear_group / groups;
    const int64_t group = linear_group - row * groups;
    const int64_t offset = row * cols + group * group_size;

    float mu = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        mu += x[offset + j];
    }
    mu /= static_cast<float>(group_size);

    float var = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        const float r = x[offset + j] - mu;
        var += r * r;
    }
    const float s = sqrtf(var / static_cast<float>(group_size) + 1.0e-6f);

    float sum_codes = 0.0f;
    unsigned char* out_row = codes + row * cols;
    for (int j = 0; j < group_size; ++j) {
        const float t = (x[offset + j] - mu) / s;
        int code = 0;
        while (code < levels - 1 && t > thresholds[code]) {
            ++code;
        }
        out_row[group * group_size + j] = static_cast<unsigned char>(code);
        sum_codes += codebook[code];
    }

    center[linear_group] = mu;
    scale[linear_group] = s;
    code_sum[linear_group] = sum_codes;
}

__global__ void quantize_activation_dequant_kernel(
    const float* __restrict__ x,
    const float* __restrict__ thresholds,
    const float* __restrict__ codebook,
    float* __restrict__ x_hat,
    int64_t rows,
    int64_t cols,
    int64_t groups,
    int group_size,
    int levels) {
    const int64_t linear_group = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_groups = rows * groups;
    if (linear_group >= total_groups) {
        return;
    }

    const int64_t row = linear_group / groups;
    const int64_t group = linear_group - row * groups;
    const int64_t offset = row * cols + group * group_size;

    float mu = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        mu += x[offset + j];
    }
    mu /= static_cast<float>(group_size);

    float var = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        const float r = x[offset + j] - mu;
        var += r * r;
    }
    const float s = sqrtf(var / static_cast<float>(group_size) + 1.0e-6f);

    float sum_codes = 0.0f;
    for (int j = 0; j < group_size; ++j) {
        const float t = (x[offset + j] - mu) / s;
        int code = 0;
        while (code < levels - 1 && t > thresholds[code]) {
            ++code;
        }
        sum_codes += codebook[code];
    }
    const float code_mean = sum_codes / static_cast<float>(group_size);

    for (int j = 0; j < group_size; ++j) {
        const float t = (x[offset + j] - mu) / s;
        int code = 0;
        while (code < levels - 1 && t > thresholds[code]) {
            ++code;
        }
        x_hat[offset + j] = mu + s * (codebook[code] - code_mean);
    }
}

}  // namespace

std::vector<torch::Tensor> quantize_activation_cuda(
    torch::Tensor x,
    torch::Tensor thresholds,
    torch::Tensor codebook,
    int64_t bits,
    int64_t group_size) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(thresholds.is_cuda(), "thresholds must be CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "codebook must be CUDA tensor");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(x.dim() == 2, "x must have shape [rows, cols]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(thresholds.scalar_type() == torch::kFloat32, "thresholds must be float32");
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat32, "codebook must be float32");
    TORCH_CHECK(x.size(1) % group_size == 0, "in_features must be divisible by group_size");

    x = x.contiguous();
    thresholds = thresholds.contiguous();
    codebook = codebook.contiguous();
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    const int64_t groups = cols / group_size;
    const int64_t packed_cols = (cols * bits + 7) / 8;
    const int levels = static_cast<int>(codebook.numel());

    auto packed = torch::zeros({rows, packed_cols}, x.options().dtype(torch::kUInt8));
    auto center = torch::empty({rows, groups}, x.options());
    auto scale = torch::empty({rows, groups}, x.options());
    auto code_sum = torch::empty({rows, groups}, x.options());

    const int threads = 128;
    const int blocks = static_cast<int>((rows * groups + threads - 1) / threads);
    quantize_activation_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        thresholds.data_ptr<float>(),
        codebook.data_ptr<float>(),
        packed.data_ptr<unsigned char>(),
        center.data_ptr<float>(),
        scale.data_ptr<float>(),
        code_sum.data_ptr<float>(),
        rows,
        cols,
        groups,
        packed_cols,
        static_cast<int>(bits),
        static_cast<int>(group_size),
        levels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed, center, scale, code_sum};
}

std::vector<torch::Tensor> quantize_activation_u8_cuda(
    torch::Tensor x,
    torch::Tensor thresholds,
    torch::Tensor codebook,
    int64_t bits,
    int64_t group_size) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(thresholds.is_cuda(), "thresholds must be CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "codebook must be CUDA tensor");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(x.dim() == 2, "x must have shape [rows, cols]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(thresholds.scalar_type() == torch::kFloat32, "thresholds must be float32");
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat32, "codebook must be float32");
    TORCH_CHECK(x.size(1) % group_size == 0, "in_features must be divisible by group_size");

    x = x.contiguous();
    thresholds = thresholds.contiguous();
    codebook = codebook.contiguous();
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    const int64_t groups = cols / group_size;
    const int levels = static_cast<int>(codebook.numel());

    auto codes = torch::empty({rows, cols}, x.options().dtype(torch::kUInt8));
    auto center = torch::empty({rows, groups}, x.options());
    auto scale = torch::empty({rows, groups}, x.options());
    auto code_sum = torch::empty({rows, groups}, x.options());

    const int threads = 128;
    const int blocks = static_cast<int>((rows * groups + threads - 1) / threads);
    quantize_activation_u8_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        thresholds.data_ptr<float>(),
        codebook.data_ptr<float>(),
        codes.data_ptr<unsigned char>(),
        center.data_ptr<float>(),
        scale.data_ptr<float>(),
        code_sum.data_ptr<float>(),
        rows,
        cols,
        groups,
        static_cast<int>(group_size),
        levels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {codes, center, scale, code_sum};
}

torch::Tensor quantize_activation_dequant_cuda(
    torch::Tensor x,
    torch::Tensor thresholds,
    torch::Tensor codebook,
    int64_t bits,
    int64_t group_size) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(thresholds.is_cuda(), "thresholds must be CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "codebook must be CUDA tensor");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(x.dim() == 2, "x must have shape [rows, cols]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(thresholds.scalar_type() == torch::kFloat32, "thresholds must be float32");
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat32, "codebook must be float32");
    TORCH_CHECK(x.size(1) % group_size == 0, "in_features must be divisible by group_size");

    x = x.contiguous();
    thresholds = thresholds.contiguous();
    codebook = codebook.contiguous();
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    const int64_t groups = cols / group_size;
    const int levels = static_cast<int>(codebook.numel());

    auto x_hat = torch::empty_like(x);
    const int threads = 128;
    const int blocks = static_cast<int>((rows * groups + threads - 1) / threads);
    quantize_activation_dequant_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        thresholds.data_ptr<float>(),
        codebook.data_ptr<float>(),
        x_hat.data_ptr<float>(),
        rows,
        cols,
        groups,
        static_cast<int>(group_size),
        levels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x_hat;
}
