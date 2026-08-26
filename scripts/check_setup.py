#!/usr/bin/env python3
"""Verify that this checkout can run Stage-2 (and optionally Stage-1)."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HiTeC.paths import DATASET_DOWNLOAD_URL

DATASETS = ("cora", "citeseer")
OPTIONAL = ("history", "photo", "computers", "fitness")
REQUIRED_DATA = ("features.pt", "hypergraph_dict.pt", "labels.pt", "texts.pt", "edge_bucket_cns.pt")
REQUIRED_EMB = ("raw_emb.pt", "augmented_emb.pt")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def check_pkg(name: str, required: bool = True) -> bool:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "")
        _ok(f"{name} {ver}".rstrip())
        return True
    except Exception as exc:
        if required:
            _fail(f"{name} not importable ({exc})")
        else:
            _warn(f"{name} not installed (only needed for Stage-1)")
        return not required


def main() -> int:
    failed = 0
    print(f"HiTeC setup check\n  root: {ROOT}\n")

    print("Python / packages")
    print(f"  [ok]   python {sys.version.split()[0]}")
    for pkg in ("torch", "yaml", "numpy", "sklearn", "tqdm"):
        if not check_pkg(pkg):
            failed += 1
    if not check_pkg("torch_scatter"):
        failed += 1
        print("         -> pip install -r requirements.txt")

    try:
        import torch
        if torch.cuda.is_available():
            _ok(f"CUDA {torch.version.cuda}  ({torch.cuda.get_device_name(0)})")
        else:
            _warn("CUDA not visible; Stage-2 will run on CPU (slow).")
    except Exception:
        pass

    stage1_ok = check_pkg("transformers", required=False) and check_pkg("peft", required=False)

    print("\nConfig")
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        _ok(f"config.yaml")
    else:
        _fail("config.yaml missing")
        failed += 1

    print("\nDatasets + Stage-1 embeddings (fast path)")
    data_root = Path(os.environ.get("HITEC_DATA_DIR", ROOT / "tahg_datasets"))
    emb_root = Path(os.environ.get("HITEC_EMB_DIR", ROOT / "emb"))
    for name in DATASETS:
        ddir = data_root / name
        missing = [f for f in REQUIRED_DATA if not (ddir / f).exists()]
        splits = ddir / "splits"
        n_splits = len(list(splits.glob("*.pt"))) if splits.exists() else 0
        if missing:
            _fail(f"{name}: missing {missing} under {ddir}")
            failed += 1
        elif n_splits < 20:
            _warn(f"{name}: found {n_splits} splits (paper uses 20)")
        else:
            _ok(f"{name}: data + {n_splits} splits")

        edir = emb_root / name
        missing_emb = [f for f in REQUIRED_EMB if not (edir / f).exists()]
        if missing_emb:
            _fail(f"{name}: missing embeddings {missing_emb} under {edir}")
            failed += 1
        else:
            _ok(f"{name}: raw_emb.pt + augmented_emb.pt")

    print("\nOther TAHGs (same layout as cora/; optional)")
    for name in OPTIONAL:
        ddir = data_root / name
        edir = emb_root / name
        data_ok = ddir.exists() and (ddir / "features.pt").exists()
        emb_ok = (edir / "raw_emb.pt").exists() and (edir / "augmented_emb.pt").exists()
        if data_ok and emb_ok:
            _ok(f"{name}")
        else:
            _warn(f"{name} not present — download {DATASET_DOWNLOAD_URL}")
            print(f"         unpack like cora: tahg_datasets/{name}/ and emb/{name}/")

    print("\nStage-1 extra deps")
    if stage1_ok:
        _ok("transformers + peft available")
    else:
        _warn("install with: pip install -r requirements.txt")

    print()
    if failed:
        print(f"Result: {failed} blocking issue(s). Fix those before training.")
        return 1
    print("Result: Stage-2 fast path is ready.")
    print("  python train.py --dataset cora --smoke --device auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
