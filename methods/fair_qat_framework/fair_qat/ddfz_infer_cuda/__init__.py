import torch  # noqa: F401  # Load PyTorch shared libraries before the extension.

try:
    from .ddfz_infer_cuda_ext import (  # type: ignore
        ddfz_linear_forward,
        ddfz_linear_forward_u8,
        pack_codes,
        quantize_activation,
        quantize_activation_dequant,
        quantize_activation_u8,
        unpack_codes,
        unpack_codes_u8,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DDFZ CUDA extension is not built. Run "
        "`python setup.py build_ext --inplace` inside "
        "`fair_qat/ddfz_infer_cuda` first."
    ) from exc

__all__ = [
    "pack_codes",
    "unpack_codes",
    "unpack_codes_u8",
    "quantize_activation",
    "quantize_activation_dequant",
    "quantize_activation_u8",
    "ddfz_linear_forward",
    "ddfz_linear_forward_u8",
]
