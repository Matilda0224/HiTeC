"""Project-root paths. Override with env vars if data lives elsewhere."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("HITEC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


ROOT = project_root()
DATA_DIR = Path(os.environ.get("HITEC_DATA_DIR", ROOT / "tahg_datasets")).expanduser().resolve()
EMB_DIR = Path(os.environ.get("HITEC_EMB_DIR", ROOT / "emb")).expanduser().resolve()
CONFIG_PATH = Path(os.environ.get("HITEC_CONFIG", ROOT / "config.yaml")).expanduser().resolve()

# Public dataset dump (same layout as tahg_datasets/cora and emb/cora).
DATASET_DOWNLOAD_URL = "https://drive.google.com/drive/folders/1tkNOf2ehJoUxvPRTxwKPGdVdiA5MXsqC?usp=sharing"


def missing_data_hint(name: str, kind: str = "dataset") -> str:
    """Tell the user to download and drop files in the Cora/CiteSeer layout."""
    if kind == "emb":
        return (
            f"Stage-1 embeddings for '{name}' were not found under {EMB_DIR / name}.\n"
            f"Download from {DATASET_DOWNLOAD_URL}\n"
            f"and place files like Cora/CiteSeer:\n"
            f"  emb/{name}/raw_emb.pt\n"
            f"  emb/{name}/augmented_emb.pt\n"
            f"Or generate them: python train.py --dataset {name} --encode_emb --train_textencoder --device auto"
        )
    return (
        f"Dataset '{name}' was not found under {DATA_DIR / name}.\n"
        f"Download from {DATASET_DOWNLOAD_URL}\n"
        f"and place files like Cora/CiteSeer:\n"
        f"  tahg_datasets/{name}/features.pt\n"
        f"  tahg_datasets/{name}/hypergraph_dict.pt\n"
        f"  tahg_datasets/{name}/labels.pt\n"
        f"  tahg_datasets/{name}/texts.pt\n"
        f"  tahg_datasets/{name}/edge_bucket_cns.pt\n"
        f"  tahg_datasets/{name}/splits/0.pt … 19.pt"
    )


def resolve_device(device_arg: str) -> str:
    """Map CLI --device to a torch device string. Accepts auto / cpu / 0 / cuda:0."""
    import torch

    raw = str(device_arg).strip().lower()
    if raw in {"cpu"}:
        return "cpu"
    if raw in {"auto", "-1", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw.isdigit():
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to CPU.")
            return "cpu"
        return f"cuda:{int(raw)}"
    if raw.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to CPU.")
            return "cpu"
        return raw
    raise ValueError(f"Unrecognized --device value: {device_arg}")


def torch_load(path, map_location=None):
    """torch.load that works on both old and new PyTorch."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
