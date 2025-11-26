"""Smoke tests: import modules and run lightweight checks.

This script performs non-interactive sanity checks on modules that are safe
to run in CI (no GUI or camera access).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
import importlib.util


def safe_import(module: str):
    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - runtime environment
        print(f"SKIP import {module}: {exc}")
        return None


def load_from_path(name: str, relpath: str):
    """Load a module directly from a filesystem path (relative to repo root)."""
    repo_root = Path(__file__).resolve().parent.parent
    full = repo_root / relpath
    if not full.exists():
        print(f"Module file not found: {full}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, str(full))
        if spec is None or spec.loader is None:
            print(f"Failed to create spec for {full}")
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        return mod
    except Exception as exc:
        print(f"Failed to load {full}: {exc}")
        return None


def main() -> None:
    np = safe_import("numpy")
    # perception: UNet predict (lane)
    lane_mod = safe_import("perception.lane_unet") or load_from_path("perception.lane_unet", "perception/lane_unet.py")
    drivable_mod = safe_import("perception.drivable_area_unet") or load_from_path("perception/drivable_area_unet.py", "perception/drivable_area_unet.py")
    if lane_mod and drivable_mod and np is not None:
        model_lane = lane_mod.build_model()
        model_drivable = drivable_mod.build_model()
        dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
        lane_mask = lane_mod.predict(model_lane, dummy_img)
        drivable_mask = drivable_mod.predict(model_drivable, dummy_img)
        print("Lane mask shape:", getattr(lane_mask, "shape", type(lane_mask)))
        print("Drivable mask shape:", getattr(drivable_mask, "shape", type(drivable_mask)))
    else:
        print("SKIPPED perception UNet checks (numpy or modules missing)")

    # policy: ConvLSTM forward
    policy_mod = safe_import("policy.convlstm_model") or load_from_path("policy.convlstm_model", "policy/convlstm_model.py")
    torch = safe_import("torch")
    if policy_mod and torch:
        model = policy_mod.ConvLSTMPolicy()
        dummy = torch.randn(1, 4, 3, 128, 128)
        out = model(dummy)
        print("Policy output shape:", out.shape)
    else:
        print("SKIPPED policy ConvLSTM check (torch or module missing)")

    # data_engine: quality metrics
    qm = safe_import("data_engine.quality_metrics") or load_from_path("data_engine.quality_metrics", "data_engine/quality_metrics.py")
    if qm and np is not None:
        q = qm.frame_quality(np.zeros((128, 128, 3), dtype=np.uint8))
        print("Quality metrics:", q)
    else:
        print("SKIPPED quality metrics check")

    # utils: dataset instantiation and config
    ds_mod = safe_import("utils.datasets") or load_from_path("utils.datasets", "utils/datasets.py")
    cfg_mod = safe_import("utils.config") or load_from_path("utils.config", "utils/config.py")
    if ds_mod and cfg_mod:
        ds = ds_mod.FrameDataset("./data")
        print("Dataset length (may be 0):", len(ds))
        print("AppConfig default:", cfg_mod.AppConfig())
    else:
        print("SKIPPED utils checks")

    print("Smoke tests completed (skips expected if optional deps missing).")


if __name__ == "__main__":
    main()
