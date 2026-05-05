from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "SkyTbac/AE_ANNS_IDX"
BIGANN100M_REPO_ID = "SkyTbac/ANNS_IDX_BigANN100M"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download ANNS index cache datasets from Hugging Face. "
            f"Default behavior downloads {REPO_ID} into ANNS_idx_cache. "
            f"When selecting bigann100m, {BIGANN100M_REPO_ID} is downloaded "
            "into ANNS_idx_cache/BigANN100M."
        )
    )
    p.add_argument(
        "--dataset",
        choices=("ae", "bigann100m"),
        default="ae",
        help=(
            "Dataset to download: 'ae' (default, SkyTbac/AE_ANNS_IDX) "
            "or 'bigann100m' (SkyTbac/ANNS_IDX_BigANN100M -> <output-dir>/BigANN100M)."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="ANNS_idx_cache",
        help="Directory to download into (default: ANNS_idx_cache).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel downloads (default: env HF_SNAPSHOT_MAX_WORKERS or 8).",
    )
    return p.parse_args()


def _max_workers(cli: int | None) -> int:
    if cli is not None:
        return max(1, cli)
    env = os.environ.get("HF_SNAPSHOT_MAX_WORKERS")
    if env is not None:
        return max(1, int(env))
    return 8


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Missing dependency 'huggingface_hub'. "
            "Install: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 2

    local_dir = Path(args.output_dir).resolve()
    repo_id = REPO_ID
    if args.dataset == "bigann100m":
        repo_id = BIGANN100M_REPO_ID
        local_dir = local_dir / "BigANN100M"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        max_workers=_max_workers(args.max_workers),
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
