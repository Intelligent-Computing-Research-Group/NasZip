from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DATASET_MAP = {
    "SIFT": "SkyTbac/ANNS_SIFT1M",
    "GIST": "SkyTbac/ANNS_GIST1M",
    "GloVe": "SkyTbac/ANNS_GloVe.twitter.27B.100d",
    "BigANN100M": "SkyTbac/ANNS_BigANN100M",
    "BigANN100M_idx": "SkyTbac/ANNS_IDX_BigANN100M",
    "MS_MARCO": "SkyTbac/ANNS_MS_MARCO_384d_8M",
    "Wiki": "SkyTbac/ANNS_Wiki_1M",
}

# Alternate CLI names -> canonical key in DATASET_MAP (output folder uses canonical name)
DATASET_ALIASES = {
    "BigANN": "BigANN100M",
}

# Subdirectory under --output-dir (default: same as DATASET_MAP key). Maps index bundles to the
# dataset folder name expected by idx_tools (e.g. ANNS_idx_cache/BigANN100M/).
DATASET_OUTPUT_SUBDIR = {
    "BigANN100M_idx": "BigANN100M",
}


def _canonical_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _valid_name_tokens() -> list[str]:
    return sorted(set(DATASET_MAP) | set(DATASET_ALIASES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download configured datasets from Hugging Face."
    )
    parser.add_argument(
        "dataset_names",
        nargs="+",
        metavar="NAME",
        help="One or more dataset short names (see repo DATASET_MAP / aliases).",
    )
    parser.add_argument(
        "--output-dir",
        default="Datasets",
        help="Root output directory for downloaded datasets (default: Datasets).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel HF file downloads (default: 8, same as huggingface_hub; "
            "override with HF_SNAPSHOT_MAX_WORKERS)."
        ),
    )
    return parser.parse_args()


def _resolve_max_workers(cli_value: int | None) -> int:
    if cli_value is not None:
        return max(1, cli_value)
    env = os.environ.get("HF_SNAPSHOT_MAX_WORKERS")
    if env is not None:
        return max(1, int(env))
    return 8  # huggingface_hub snapshot_download default


def download_one(canonical_name: str, output_dir: str, *, max_workers: int) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
        ) from exc

    repo_id = DATASET_MAP[canonical_name]
    subdir = DATASET_OUTPUT_SUBDIR.get(canonical_name, canonical_name)
    local_dir = Path(output_dir) / subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        max_workers=max_workers,
    )
    return local_dir


def main() -> int:
    args = parse_args()
    max_workers = _resolve_max_workers(args.max_workers)
    resolved: list[tuple[str, str]] = []
    for raw in _dedupe_preserve_order(args.dataset_names):
        canonical = _canonical_name(raw)
        if canonical not in DATASET_MAP:
            print(
                f"Unknown dataset name: {raw!r}. Valid names: {', '.join(_valid_name_tokens())}",
                file=sys.stderr,
            )
            return 2
        resolved.append((raw, canonical))

    exit_code = 0
    for raw, canonical in resolved:
        label = f"{raw} ({canonical})" if raw != canonical else canonical
        print(f"========== Downloading: {label} ==========")
        try:
            local_path = download_one(
                canonical, args.output_dir, max_workers=max_workers
            )
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        print(
            f"\nDownload completed: {canonical} ({DATASET_MAP[canonical]}) -> {local_path}"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
