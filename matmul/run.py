#!/usr/bin/env python3
"""
Matmul kernel harness for local benchmarking & debugging (Nsight-friendly).

Key features
------------
- JIT-compiles any matmul CUDA file via torch.utils.cpp_extension.load_inline.
- Deterministic test generation with torch; validates vs torch.matmul.
- Benchmarks with repeat timing; reports max error, best runtime, GFLOPs.
- Places all build artefacts (so, ptx, cubin, intermediates) under matmul/build_artifacts
  for easy access with cuobjdump/nvdisasm and Nsight integrations.

Usage examples
--------------
python run.py --list
python run.py --kernel t4/matmul.cu --gpu-id 0 --preset quick
python run.py --kernel t4/923_14_GFLOP_matmul_shmem.cu --gpu-id 1 --preset tensara --repeats 5 --arch sm_80
python run.py --kernel t4/matmul.cu --gpu-id 0 --sizes 1024x1024x1024 2048x2048x1024 --keep
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.cpp_extension import load_inline


THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_KERNEL_DIR = THIS_DIR
BUILD_DIR = THIS_DIR / "build_artifacts"
KEEP_DIR = BUILD_DIR / "nvcc_keep"

PRESETS = {
    "quick": [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ],
    "tensara": [
        (1024, 1024, 1024),
        (2048, 2048, 1024),
        (4096, 4096, 4096),
    ],
}


def list_available_kernels() -> List[pathlib.Path]:
    # recurse to allow future targets (e.g., different GPUs) in subdirs
    return sorted(DEFAULT_KERNEL_DIR.rglob("*.cu"))


def parse_shape(raw: str) -> Tuple[int, int, int]:
    for sep in ("x", "X", ","):
        if sep in raw:
            parts = raw.split(sep)
            break
    else:
        raise ValueError(f"Could not parse shape '{raw}'. Expected MxKxN.")
    if len(parts) != 3:
        raise ValueError(f"Shape '{raw}' must have 3 dimensions (M,K,N).")
    m, k, n = (int(p) for p in parts)
    return m, k, n


def parse_shapes(values: Sequence[str]) -> List[Tuple[int, int, int]]:
    return [parse_shape(v) for v in values]


def ensure_dirs(keep: bool):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if keep:
        KEEP_DIR.mkdir(parents=True, exist_ok=True)


def _extract_arch_ver(arch: str) -> str:
    """Extract the version digits from an arch string (e.g., 'sm_120' -> '120')."""
    return arch.replace("sm", "").replace("_", "")


def _sm_to_torch_arch(sm: str) -> str:
    """Convert 'sm80'/'sm_80' -> '8.0' format expected by TORCH_CUDA_ARCH_LIST."""
    digits = _extract_arch_ver(sm)
    if not digits:
        raise ValueError(f"Could not parse CUDA arch version from '{sm}'")
        
    if len(digits) <= 2:
        major, minor = digits[:-1] or digits, digits[-1]
    else:  # e.g., sm120 -> major=12, minor=0
        major, minor = digits[:-1], digits[-1]
    return f"{int(major)}.{int(minor)}"


def detect_arch(device: torch.device) -> str:
    """Detects architecture and returns standard nvcc format (e.g. sm_90)."""
    major, minor = torch.cuda.get_device_capability(device=device)
    # Return 'sm_XY' which is generally safer/standard for nvcc than 'smXY'
    return f"sm_{major}{minor}"


def choose_arch(user_arch: str | None, device: torch.device) -> str:
    """Pick an arch supported by the installed CUDA toolkit.

    Trusts the user's input if provided. Otherwise, detects device arch.
    """
    if user_arch:
        # User is responsible for correct formatting (e.g. sm_120 vs sm120)
        return user_arch

    detected = detect_arch(device)

    try:
        torch_arch_list = torch.cuda.get_arch_list()
    except Exception:
        torch_arch_list = []

    # Clean the torch list to compare properly
    # (Torch often returns ['sm_90', 'sm_80'], which matches our detect_arch format)
    if torch_arch_list and detected not in torch_arch_list:
        # Fallback logic: sort available archs and pick the highest one
        # We use _extract_arch_ver to sort numerically
        def sort_key(a):
            try:
                return int(_extract_arch_ver(a))
            except ValueError:
                return -1

        supported = sorted(torch_arch_list, key=sort_key)
        fallback = supported[-1] if supported else detected
        
        if fallback != detected:
            print(
                f"[info] Detected arch {detected} not in torch arch list {torch_arch_list}; "
                f"falling back to {fallback}. Override with --arch to change."
            )
            return fallback

    return detected


def ensure_torch_arch_env(sm_arch: str):
    """Set TORCH_CUDA_ARCH_LIST if the user hasn't provided one.

    This prevents torch.utils.cpp_extension from appending unsupported archs
    (e.g., sm120) when NVCC doesn't yet recognize the latest GPU.
    """
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return
    os.environ["TORCH_CUDA_ARCH_LIST"] = _sm_to_torch_arch(sm_arch)


def build_extension(kernel_path: pathlib.Path, arch: str | None, keep: bool, verbose: bool = False):
    cuda_src = kernel_path.read_text()
    binding_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void solution(const float* input_a, const float* input_b, float* output_c,
                         size_t m, size_t n, size_t k);

torch::Tensor launch(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda(), "input A must be on CUDA");
  TORCH_CHECK(b.is_cuda(), "input B must be on CUDA");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same device");
  TORCH_CHECK(a.scalar_type() == torch::kFloat, "A must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat, "B must be float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "A.cols must equal B.rows");


  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  const auto m = a_c.size(0);
  const auto k = a_c.size(1);
  const auto n = b_c.size(1);

  auto out = torch::zeros({m, n}, a_c.options());

  solution(static_cast<float*>(a_c.data_ptr<float>()),
           static_cast<float*>(b_c.data_ptr<float>()),
           static_cast<float*>(out.data_ptr<float>()),
           static_cast<size_t>(m),
           static_cast<size_t>(n),
           static_cast<size_t>(k));

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch, "Launch matmul kernel (A @ B)");
}
"""
    name = f"matmul_ext_{kernel_path.stem}"

    # Standard flags
    extra_cuda_cflags = ["-O3", "-lineinfo", "-Xptxas", "-v"]
    
    if arch:
        ver = _extract_arch_ver(arch)
        extra_cuda_cflags.append(f"-gencode=arch=compute_{ver},code={arch}")
    
    if keep:
        # --keep retains .ptx, .cubin, etc.
        # --source-in-ptx interleaves your C++ code into the PTX file
        extra_cuda_cflags += ["--keep", f"--keep-dir={KEEP_DIR}", "--source-in-ptx"]

    ext = load_inline(
        name=name,
        cpp_sources=binding_src,
        cuda_sources=cuda_src,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        build_directory=str(BUILD_DIR),
        verbose=verbose,
    )

    # AUTO-DISASSEMBLE: Generate readable .sass from the .cubin
    if keep:
        import subprocess
        
        # Torch typically writes the CUDA source to 'cuda.cu', so nvcc outputs 'cuda.cubin'
        cubin_path = KEEP_DIR / "cuda.cubin"
        
        # Fallback: find any cubin if the default name changed
        if not cubin_path.exists():
            candidates = list(KEEP_DIR.glob("*.cubin"))
            if candidates:
                cubin_path = candidates[0]

        if cubin_path.exists():
            sass_path = KEEP_DIR / f"{kernel_path.stem}.sass"
            print(f"[info] Disassembling {cubin_path.name} -> {sass_path.name}...")
            # -g for source interleaving (requires -lineinfo which we have)
            # --print-code-hotness shows instruction execution frequency if you have profile data (optional)
            cmd = f"nvdisasm -g {cubin_path} > {sass_path}"
            subprocess.run(cmd, shell=True, check=False)
        else:
            print("[warning] Could not find .cubin file to disassemble.")

    return ext


def run_case(ext, shape: Tuple[int, int, int], repeats: int, tol: float, device: torch.device):
    m, k, n = shape
    a = torch.randn((m, k), device=device, dtype=torch.float32)
    b = torch.randn((k, n), device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    out = ext.launch(a, b)
    torch.cuda.synchronize()
    ref = a @ b
    max_err = (out - ref).abs().max().item()
    status = "PASS" if max_err <= tol else "FAIL"

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ext.launch(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best = min(times)
    gflops = (2.0 * m * n * k) / 1e9 / best
    return {
        "shape": shape,
        "status": status,
        "max_err": max_err,
        "best_ms": best * 1e3,
        "gflops": gflops,
    }


def choose_shapes(args) -> List[Tuple[int, int, int]]:
    if args.sizes:
        return parse_shapes(args.sizes)
    return PRESETS[args.preset]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile and run matmul kernels via PyTorch.")
    parser.add_argument("--kernel", default="t4/matmul.cu", help="Kernel path relative to matmul/ or absolute.")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="quick", help="Predefined shape set.")
    parser.add_argument("--sizes", nargs="*", help="Override shapes, e.g. 1024x1024x1024 2048x2048x1024.")
    parser.add_argument("--repeats", type=int, default=10, help="Benchmark repetitions per shape.")
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Max allowed |diff| for PASS.")
    parser.add_argument("--gpu-id", type=int, required=True, help="CUDA device ordinal to use (e.g., 0).")
    parser.add_argument("--list", action="store_true", help="List available kernels and exit.")
    parser.add_argument("--verbose", action="store_true", help="Show compilation output.")
    parser.add_argument("--arch", help="Explicit SM target, e.g. sm_80. Defaults to detected device.")
    parser.add_argument("--keep", action="store_true", help="Keep NVCC intermediates (ptx/cubin) in build_artifacts/nvcc_keep.")
    args = parser.parse_args(argv)

    if args.list:
        for path in list_available_kernels():
            print(path.relative_to(DEFAULT_KERNEL_DIR))
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Please select a CUDA device.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise SystemExit(f"Invalid gpu-id {args.gpu_id}; available device count: {torch.cuda.device_count()}.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    arch = choose_arch(args.arch, device)
    ensure_torch_arch_env(arch)

    shapes = choose_shapes(args)

    kernel_path = pathlib.Path(args.kernel)
    if not kernel_path.is_absolute():
        kernel_path = DEFAULT_KERNEL_DIR / kernel_path
    if not kernel_path.exists():
        raise SystemExit(f"Kernel file not found: {kernel_path}")

    ensure_dirs(args.keep)
    ext = build_extension(kernel_path, arch=arch, keep=args.keep, verbose=args.verbose)

    print(f"Using kernel: {kernel_path.relative_to(DEFAULT_KERNEL_DIR)}")
    print(f"Device: {device} | Arch: {arch}")
    print(f"Seed: {args.seed}")
    print(f"Shapes: {', '.join(f'{m}x{k}x{n}' for m, k, n in shapes)}")
    print(f"Build artifacts: {BUILD_DIR}")
    if args.keep:
        print(f"NVCC keep dir (ptx/cubin): {KEEP_DIR}")
    print("-" * 78)
    print(f"{'Shape':>18} | {'Status':>6} | {'Max Err':>10} | {'Best ms':>8} | {'GFLOPs':>10}")
    print("-" * 78)

    for shape in shapes:
        result = run_case(ext, shape, repeats=args.repeats, tol=args.tol, device=device)
        m, k, n = result["shape"]
        print(
            f"{m}x{k}x{n: <4} | "
            f"{result['status']:>6} | "
            f"{result['max_err']:10.4e} | "
            f"{result['best_ms']:8.3f} | "
            f"{result['gflops']:10.2f}"
        )
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())