import importlib
import sys
from typing import Optional


def check_import(name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return str(version)
    except Exception as e:
        print(f"[WARN] Could not import {name}: {e}")
        return None


def check_torch_cuda() -> None:
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(f"[ERROR] torch check failed: {e}")


def main() -> int:
    print("=== CircuitKit Environment Validation ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {sys.platform}")

    # Core
    check_torch_cuda()
    transformers_ver = check_import("transformers")
    if transformers_ver:
        print(f"transformers: {transformers_ver}")

    # Optional/internal
    dlbactrace_ver = check_import("dlbactrace")
    if dlbactrace_ver:
        print(f"dlbactrace: {dlbactrace_ver}")
    else:
        print("[INFO] dlbactrace not found; continuing without it.")

    # CircuitKit import
    ck_ver = check_import("circuitkit")
    if ck_ver is not None:
        print("circuitkit import: OK")
    else:
        print("[ERROR] circuitkit import failed")
        return 1

    print("Validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
