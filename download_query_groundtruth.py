from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "SkyTbac/AE_ANNS_query_groundtruth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Download {REPO_ID} from Hugging Face into the given directory "
        f"(default: Datasets). Use --subset to fetch a single subfolder only."
    )
    p.add_argument(
        "--output-dir",
        default="Datasets",
        help="Directory to download into (default: Datasets).",
    )
    p.add_argument(
        "--subset",
        default=None,
        metavar="NAME",
        help=(
            "Only download this top-level folder from the dataset repo "
            "(e.g. SIFT1M, GIST1M, GloVe, BigANN100M, ANNS_Wiki_1M, MS_MARCO_384d_8M). "
            "Omit to download the full repository."
        ),
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
    local_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [f"{args.subset}/**"] if args.subset else None

    print(f"Downloading {REPO_ID} -> {local_dir}" + (f"  (subset: {args.subset})" if args.subset else ""))
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=_max_workers(args.max_workers),
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
