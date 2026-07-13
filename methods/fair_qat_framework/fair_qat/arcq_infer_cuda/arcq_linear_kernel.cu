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

__global__ void arcq_linear_forward_kernel(
    const unsigned char* __restrict__ packed_x,
    const float* __restrict__ x_center,
    const float* __restrict__ x_scale,
    const float* __restrict__ x_code_sum,
    const unsigned char* __restrict__ packed_w,
    const float* __restrict__ w_center,
    const float* __restrict__ w_scale,
    const float* __restrict__ w_code_sum,
    const float* __restrict__ product_table,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t groups,
    int64_t packed_cols,
    int bits,
    int group_size,
    int levels,
    bool has_bias) {
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * out_features;
    if (linear >= total) {
        return;
    }

    const int64_t row = linear / out_features;
    const int64_t out = linear - row * out_features;
    const unsigned char* x_row = packed_x + row * packed_cols;
    const unsigned char* w_row = packed_w + out * packed_cols;
    double acc = has_bias ? static_cast<double>(bias[out]) : 0.0;

    for (int64_t g = 0; g < groups; ++g) {
        double cross = 0.0;
        const int64_t base = g * group_size;
        for (int j = 0; j < group_size; ++j) {
            const int x_code = get_packed_code_u8(x_row, base + j, bits);
            const int w_code = get_packed_code_u8(w_row, base + j, bits);
            cross += static_cast<double>(product_table[x_code * levels + w_code]);
        }
        const int64_t x_meta = row * groups + g;
        const int64_t w_meta = out * groups + g;
        const double mu_term = static_cast<double>(group_size)
            * static_cast<double>(x_center[x_meta])
            * static_cast<double>(w_center[w_meta]);
        const double residual_term = static_cast<double>(x_scale[x_meta])
            * static_cast<double>(w_scale[w_meta])
            * (cross - (
                static_cast<double>(x_code_sum[x_meta])
                * static_cast<double>(w_code_sum[w_meta])
            ) / static_cast<double>(group_size));
        acc += mu_term + residual_term;
    }

    y[linear] = static_cast<float>(acc);
}

__global__ void arcq_linear_forward_u8_kernel(
    const unsigned char* __restrict__ x_codes,
    const float* __restrict__ x_center,
    const float* __restrict__ x_scale,
    const float* __restrict__ x_code_sum,
    const unsigned char* __restrict__ w_codes,
    const float* __restrict__ w_center,
    const float* __restrict__ w_scale,
    const float* __restrict__ w_code_sum,
    const float* __restrict__ product_table,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t groups,
    int group_size,
    int levels,
    bool has_bias) {
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * out_features;
    if (linear >= total) {
        return;
    }

    const int64_t row = linear / out_features;
    const int64_t out = linear - row * out_features;
    const unsigned char* x_row = x_codes + row * in_features;
    const unsigned char* w_row = w_codes + out * in_features;
    double acc = has_bias ? static_cast<double>(bias[out]) : 0.0;

    for (int64_t g = 0; g < groups; ++g) {
        double cross = 0.0;
        const int64_t base = g * group_size;
        for (int j = 0; j < group_size; ++j) {
            const int x_code = static_cast<int>(x_row[base + j]);
            const int w_code = static_cast<int>(w_row[base + j]);
            cross += static_cast<double>(product_table[x_code * levels + w_code]);
        }
        const int64_t x_meta = row * groups + g;
        const int64_t w_meta = out * groups + g;
        const double mu_term = static_cast<double>(group_size)
            * static_cast<double>(x_center[x_meta])
            * static_cast<double>(w_center[w_meta]);
        const double residual_term = static_cast<double>(x_scale[x_meta])
            * static_cast<double>(w_scale[w_meta])
            * (cross - (
                static_cast<double>(x_code_sum[x_meta])
                * static_cast<double>(w_code_sum[w_meta])
            ) / static_cast<double>(group_size));
        acc += mu_term + residual_term;
    }

    y[linear] = static_cast<float>(acc);
}

__global__ void arcq_linear_forward_u8_warp_kernel(
    const unsigned char* __restrict__ x_codes,
    const float* __restrict__ x_center,
    const float* __restrict__ x_scale,
    const float* __restrict__ x_code_sum,
    const unsigned char* __restrict__ w_codes,
    const float* __restrict__ w_center,
    const float* __restrict__ w_scale,
    const float* __restrict__ w_code_sum,
    const float* __restrict__ product_table,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t groups,
    int group_size,
    int levels,
    bool has_bias) {
    constexpr int kWarpSize = 32;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int64_t linear = (static_cast<int64_t>(blockIdx.x) * warps_per_block) + warp_in_block;
    const int64_t total = rows * out_features;
    if (linear >= total) {
        return;
    }

    const int64_t row = linear / out_features;
    const int64_t out = linear - row * out_features;
    const unsigned char* x_row = x_codes + row * in_features;
    const unsigned char* w_row = w_codes + out * in_features;
    float acc = has_bias ? bias[out] : 0.0f;
    const unsigned int mask = 0xffffffffu;

    for (int64_t g = 0; g < groups; ++g) {
        float cross = 0.0f;
        const int64_t base = g * group_size;
        for (int j = lane; j < group_size; j += kWarpSize) {
            const int x_code = static_cast<int>(x_row[base + j]);
            const int w_code = static_cast<int>(w_row[base + j]);
            cross += product_table[x_code * levels + w_code];
        }

        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            cross += __shfl_down_sync(mask, cross, offset);
        }

        if (lane == 0) {
            const int64_t x_meta = row * groups + g;
            const int64_t w_meta = out * groups + g;
            const float mu_term = static_cast<float>(group_size)
                * x_center[x_meta]
                * w_center[w_meta];
            const float residual_term = x_scale[x_meta]
                * w_scale[w_meta]
                * (cross - (x_code_sum[x_meta] * w_code_sum[w_meta]) / static_cast<float>(group_size));
            acc += mu_term + residual_term;
        }
    }

    if (lane == 0) {
        y[linear] = acc;
    }
}

}  // namespace

torch::Tensor arcq_linear_forward_cuda(
    torch::Tensor packed_x,
    torch::Tensor x_center,
    torch::Tensor x_scale,
    torch::Tensor x_code_sum,
    torch::Tensor packed_w,
    torch::Tensor w_center,
    torch::Tensor w_scale,
    torch::Tensor w_code_sum,
    torch::Tensor product_table,
    c10::optional<torch::Tensor> bias,
    int64_t bits,
    int64_t group_size,
    int64_t in_features,
    int64_t out_features) {
    TORCH_CHECK(packed_x.is_cuda(), "packed_x must be CUDA tensor");
    TORCH_CHECK(packed_w.is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(packed_x.scalar_type() == torch::kUInt8, "packed_x must be uint8");
    TORCH_CHECK(packed_w.scalar_type() == torch::kUInt8, "packed_w must be uint8");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
    TORCH_CHECK(in_features % group_size == 0, "in_features must be divisible by group_size");
    TORCH_CHECK(product_table.scalar_type() == torch::kFloat32, "product_table must be float32");

    const int64_t rows = packed_x.size(0);
    const int64_t packed_cols = packed_x.size(1);
    const int64_t groups = in_features / group_size;
    const int64_t levels = product_table.size(0);
    auto y = torch::empty({rows, out_features}, x_center.options().dtype(torch::kFloat32));
    auto x_center_c = x_center.to(torch::kFloat32).contiguous();
    auto x_scale_c = x_scale.to(torch::kFloat32).contiguous();
    auto x_code_sum_c = x_code_sum.to(torch::kFloat32).contiguous();
    auto w_center_c = w_center.to(torch::kFloat32).contiguous();
    auto w_scale_c = w_scale.to(torch::kFloat32).contiguous();
    auto w_code_sum_c = w_code_sum.to(torch::kFloat32).contiguous();
    auto product_table_c = product_table.to(torch::kFloat32).contiguous();

    const float* bias_ptr = nullptr;
    bool has_bias = false;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined() && bias.value().numel() > 0) {
        bias_contig = bias.value().to(torch::kFloat32).contiguous();
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be CUDA tensor");
        bias_ptr = bias_contig.data_ptr<float>();
        has_bias = true;
    }

    const int threads = 128;
    const int64_t total = rows * out_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    arcq_linear_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed_x.data_ptr<unsigned char>(),
        x_center_c.data_ptr<float>(),
        x_scale_c.data_ptr<float>(),
        x_code_sum_c.data_ptr<float>(),
        packed_w.data_ptr<unsigned char>(),
        w_center_c.data_ptr<float>(),
        w_scale_c.data_ptr<float>(),
        w_code_sum_c.data_ptr<float>(),
        product_table_c.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        rows,
        out_features,
        in_features,
        groups,
        packed_cols,
        static_cast<int>(bits),
        static_cast<int>(group_size),
        static_cast<int>(levels),
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor arcq_linear_forward_u8_cuda(
    torch::Tensor x_codes,
    torch::Tensor x_center,
    torch::Tensor x_scale,
    torch::Tensor x_code_sum,
    torch::Tensor w_codes,
    torch::Tensor w_center,
    torch::Tensor w_scale,
    torch::Tensor w_code_sum,
    torch::Tensor product_table,
    c10::optional<torch::Tensor> bias,
    int64_t group_size) {
    TORCH_CHECK(x_codes.is_cuda(), "x_codes must be CUDA tensor");
    TORCH_CHECK(w_codes.is_cuda(), "w_codes must be CUDA tensor");
    TORCH_CHECK(x_codes.scalar_type() == torch::kUInt8, "x_codes must be uint8");
    TORCH_CHECK(w_codes.scalar_type() == torch::kUInt8, "w_codes must be uint8");
    TORCH_CHECK(x_codes.dim() == 2, "x_codes must have shape [rows, in_features]");
    TORCH_CHECK(w_codes.dim() == 2, "w_codes must have shape [out_features, in_features]");
    TORCH_CHECK(x_codes.size(1) == w_codes.size(1), "x_codes and w_codes in_features must match");
    TORCH_CHECK(x_codes.size(1) % group_size == 0, "in_features must be divisible by group_size");
    TORCH_CHECK(product_table.scalar_type() == torch::kFloat32, "product_table must be float32");

    const int64_t rows = x_codes.size(0);
    const int64_t in_features = x_codes.size(1);
    const int64_t out_features = w_codes.size(0);
    const int64_t groups = in_features / group_size;
    const int64_t levels = product_table.size(0);
    auto y = torch::empty({rows, out_features}, x_center.options().dtype(torch::kFloat32));
    auto x_center_c = x_center.to(torch::kFloat32).contiguous();
    auto x_scale_c = x_scale.to(torch::kFloat32).contiguous();
    auto x_code_sum_c = x_code_sum.to(torch::kFloat32).contiguous();
    auto w_center_c = w_center.to(torch::kFloat32).contiguous();
    auto w_scale_c = w_scale.to(torch::kFloat32).contiguous();
    auto w_code_sum_c = w_code_sum.to(torch::kFloat32).contiguous();
    auto product_table_c = product_table.to(torch::kFloat32).contiguous();
    auto x_codes_c = x_codes.contiguous();
    auto w_codes_c = w_codes.contiguous();

    const float* bias_ptr = nullptr;
    bool has_bias = false;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined() && bias.value().numel() > 0) {
        bias_contig = bias.value().to(torch::kFloat32).contiguous();
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be CUDA tensor");
        bias_ptr = bias_contig.data_ptr<float>();
        has_bias = true;
    }

    const int threads = 128;
    const int warps_per_block = threads / 32;
    const int64_t total = rows * out_features;
    const int blocks = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    arcq_linear_forward_u8_warp_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_codes_c.data_ptr<unsigned char>(),
        x_center_c.data_ptr<float>(),
        x_scale_c.data_ptr<float>(),
        x_code_sum_c.data_ptr<float>(),
        w_codes_c.data_ptr<unsigned char>(),
        w_center_c.data_ptr<float>(),
        w_scale_c.data_ptr<float>(),
        w_code_sum_c.data_ptr<float>(),
        product_table_c.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        rows,
        out_features,
        in_features,
        groups,
        static_cast<int>(group_size),
        static_cast<int>(levels),
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
