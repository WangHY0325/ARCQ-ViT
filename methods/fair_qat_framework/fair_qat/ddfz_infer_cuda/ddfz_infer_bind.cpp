#include <torch/extension.h>

torch::Tensor pack_codes_cuda(torch::Tensor codes, int64_t bits);
torch::Tensor unpack_codes_cuda(torch::Tensor packed, int64_t bits, int64_t length);
torch::Tensor unpack_codes_u8_cuda(torch::Tensor packed, int64_t bits, int64_t length);

std::vector<torch::Tensor> quantize_activation_cuda(
    torch::Tensor x,
    torch::Tensor thresholds,
    torch::Tensor codebook,
    int64_t bits,
    int64_t group_size);

std::vector<torch::Tensor> quantize_activation_u8_cuda(
    torch::Tensor x,
    torch::Tensor thresholds,
    torch::Tensor codebook,
    int64_t bits,
    int64_t group_size);

torch::Tensor quantize_activation_dequant_cuda(
    torch::Tensor x,
    torch::Tensor thresholds,
    torch::Tensor codebook,
    int64_t bits,
    int64_t group_size);

torch::Tensor ddfz_linear_forward_cuda(
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
    int64_t out_features);

torch::Tensor ddfz_linear_forward_u8_cuda(
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
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_codes", &pack_codes_cuda, "Pack low-bit DDFZ codes");
    m.def("unpack_codes", &unpack_codes_cuda, "Unpack low-bit DDFZ codes");
    m.def("unpack_codes_u8", &unpack_codes_u8_cuda, "Unpack low-bit DDFZ codes to uint8");
    m.def("quantize_activation", &quantize_activation_cuda, "Quantize DDFZ activations");
    m.def("quantize_activation_u8", &quantize_activation_u8_cuda, "Quantize DDFZ activations to uint8 codes");
    m.def("quantize_activation_dequant", &quantize_activation_dequant_cuda, "Quantize and dequantize DDFZ activations");
    m.def("ddfz_linear_forward", &ddfz_linear_forward_cuda, "DDFZ packed linear forward");
    m.def("ddfz_linear_forward_u8", &ddfz_linear_forward_u8_cuda, "DDFZ uint8-code linear forward");
}
