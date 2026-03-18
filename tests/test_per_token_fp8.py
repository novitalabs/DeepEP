"""
Standalone unit test for per-token FP8 quantization logic.
Tests the reference implementation without requiring NVSHMEM or multi-GPU setup.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils import per_token_cast_to_fp8, per_token_cast_to_fp8_pertok, per_token_cast_back, calc_diff


def test_per_token_cast_roundtrip():
    """Test that per-token FP8 cast -> cast_back recovers original data within tolerance."""
    torch.manual_seed(42)
    for m in [1, 16, 128]:
        for n in [2048, 5120, 7168, 8192]:
            x = torch.randn(m, n, dtype=torch.bfloat16, device='cuda')
            x_fp8, x_scales = per_token_cast_to_fp8_pertok(x)

            assert x_fp8.shape == (m, n), f"FP8 shape mismatch: {x_fp8.shape}"
            assert x_fp8.dtype == torch.float8_e4m3fn
            assert x_scales.shape == (m, 1), f"Scale shape mismatch: {x_scales.shape}"
            assert x_scales.dtype == torch.float32

            # Cast back
            x_recovered = per_token_cast_back(x_fp8, x_scales)
            assert x_recovered.dtype == torch.bfloat16
            assert x_recovered.shape == (m, n)

            diff = calc_diff(x, x_recovered)
            assert diff < 5e-3, f"Per-token roundtrip error too large: {diff} for shape ({m}, {n})"

    print("PASSED: test_per_token_cast_roundtrip")


def test_per_token_vs_per128_error():
    """Verify that per-token has higher error than per-128 (as expected)."""
    torch.manual_seed(42)
    x = torch.randn(64, 7168, dtype=torch.bfloat16, device='cuda')

    # Per-128-channel
    fp8_128, scales_128 = per_token_cast_to_fp8(x)
    x_back_128 = per_token_cast_back(fp8_128, scales_128)
    diff_128 = calc_diff(x, x_back_128)

    # Per-token
    fp8_pt, scales_pt = per_token_cast_to_fp8_pertok(x)
    x_back_pt = per_token_cast_back(fp8_pt, scales_pt)
    diff_pt = calc_diff(x, x_back_pt)

    assert diff_pt >= diff_128, \
        f"Per-token error ({diff_pt}) should be >= per-128 error ({diff_128})"
    assert diff_128 < 5e-4, f"Per-128 error too large: {diff_128}"
    assert diff_pt < 5e-3, f"Per-token error too large: {diff_pt}"

    print(f"PASSED: test_per_token_vs_per128_error (per-128: {diff_128:.2e}, per-token: {diff_pt:.2e})")


def test_per_token_scale_shape():
    """Verify scale shapes for per-token vs per-128."""
    x = torch.randn(32, 7168, dtype=torch.bfloat16, device='cuda')

    _, scales_128 = per_token_cast_to_fp8(x)
    assert scales_128.shape == (32, 7168 // 128), f"Per-128 scale shape wrong: {scales_128.shape}"

    _, scales_pt = per_token_cast_to_fp8_pertok(x)
    assert scales_pt.shape == (32, 1), f"Per-token scale shape wrong: {scales_pt.shape}"

    print("PASSED: test_per_token_scale_shape")


def test_cast_back_compatibility():
    """Verify that per_token_cast_back works with both scale shapes via broadcasting."""
    torch.manual_seed(42)
    x = torch.randn(16, 7168, dtype=torch.bfloat16, device='cuda')

    # Per-token scales: [m, 1]
    fp8_pt, scales_pt = per_token_cast_to_fp8_pertok(x)
    x_back = per_token_cast_back(fp8_pt, scales_pt)
    assert x_back.shape == x.shape
    diff = calc_diff(x, x_back)
    assert diff < 5e-3, f"Cast back with per-token scales failed: {diff}"

    # Per-128 scales: [m, hidden//128]
    fp8_128, scales_128 = per_token_cast_to_fp8(x)
    x_back_128 = per_token_cast_back(fp8_128, scales_128)
    assert x_back_128.shape == x.shape
    diff_128 = calc_diff(x, x_back_128)
    assert diff_128 < 1e-4, f"Cast back with per-128 scales failed: {diff_128}"

    print("PASSED: test_cast_back_compatibility")


def test_uniform_values():
    """Test with uniform values where per-token should be exact (same amax everywhere)."""
    m, n = 8, 7168
    x = torch.ones(m, n, dtype=torch.bfloat16, device='cuda') * 3.0
    fp8, scales = per_token_cast_to_fp8_pertok(x)
    x_back = per_token_cast_back(fp8, scales)
    diff = calc_diff(x, x_back)
    # With uniform values, per-token should be as accurate as per-128
    assert diff < 1e-5, f"Uniform values roundtrip error too large: {diff}"

    print("PASSED: test_uniform_values")


if __name__ == '__main__':
    test_per_token_cast_roundtrip()
    test_per_token_vs_per128_error()
    test_per_token_scale_shape()
    test_cast_back_compatibility()
    test_uniform_values()
    print("\nAll per-token FP8 tests passed!")
