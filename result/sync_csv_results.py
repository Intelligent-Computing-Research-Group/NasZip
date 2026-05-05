import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"


DATASET_NAME_ALIASES = {
    "sift": "SIFT",
    "sift1m": "SIFT",
    "gist": "GIST",
    "gist1m": "GIST",
    "glove": "GloVe",
    "glove1_2m": "GloVe",
    "ms_marco": "MS_MARCO",
    "msmarco": "MS_MARCO",
    "msmacro8m": "MS_MARCO",
    "wiki": "Wiki",
    "wiki1m": "Wiki",
    "bigann": "BigANN",
    "bigann100m": "BigANN",
    "sift100m": "BigANN",
}


def canonical_dataset_name(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("-", "").replace("_", "")
    return DATASET_NAME_ALIASES.get(normalized, text)


def get_first(row, *keys, default=None):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def normalize_k(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def replace_rows(rows, predicate, new_rows):
    first_index = next((idx for idx, row in enumerate(rows) if predicate(row)), len(rows))
    kept_rows = [row for row in rows if not predicate(row)]
    return kept_rows[:first_index] + new_rows + kept_rows[first_index:]


def format_qps(value):
    """写入 CSV 的 QPS：纯数字字符串，无千分位逗号，避免 csv 模块对字段加引号。"""
    if value is None or value == "":
        return value
    try:
        text = str(value).replace(",", "").strip()
        return f"{float(text):.2f}"
    except (TypeError, ValueError):
        return value


def update_latency_breakdown():
    src = ROOT / "NasZip_overall.csv"
    dst = DATA_DIR / "latency_breakdown.csv"
    if not src.exists() or not dst.exists():
        return

    src_rows = read_csv(src)
    dst_rows = read_csv(dst)

    src_by_dataset = {
        canonical_dataset_name(get_first(row, "dataset_name", "dataset")): row
        for row in src_rows
    }

    for row in dst_rows:
        if row.get("design_name") != "MY":
            continue
        dataset = canonical_dataset_name(row.get("dataset_name"))
        src_row = src_by_dataset.get(dataset)
        if src_row is None:
            continue
        row["latency_10us"] = get_first(src_row, "latency", "latency_10us", default=row["latency_10us"])
        row["dist_cal"] = get_first(
            src_row, "Distance Calculation", "dist_cal", default=row["dist_cal"]
        )
        row["nbr_list_fetch"] = get_first(
            src_row, "Neighbor List Fetch", "nbr_list_fetch", default=row["nbr_list_fetch"]
        )
        row["parital_res_processing"] = src_row.get(
            "Partial Result Processing", row["parital_res_processing"]
        )

    write_csv(dst, dst_rows)


def update_overall():
    src = ROOT / "NasZip_overall.csv"
    dst = DATA_DIR / "overall.csv"
    if not src.exists() or not dst.exists():
        return

    src_rows = read_csv(src)
    dst_rows = read_csv(dst)
    src_by_dataset = {
        canonical_dataset_name(get_first(row, "dataset_name", "dataset")): row
        for row in src_rows
    }

    for row in dst_rows:
        if row.get("design_name") != "NasZip":
            continue
        dataset = canonical_dataset_name(row.get("dataset"))
        src_row = src_by_dataset.get(dataset)
        if src_row is None:
            continue
        row["QPS"] = format_qps(get_first(src_row, "qps", "QPS", default=row["QPS"]))

    write_csv(dst, dst_rows)


def update_overall_scaling():
    src = ROOT / "NasZip_overall_hp.csv"
    dst = DATA_DIR / "overall_scaling.csv"
    if not src.exists() or not dst.exists():
        return

    src_rows = read_csv(src)
    dst_rows = read_csv(dst)
    src_by_dataset_and_k = {
        (
            canonical_dataset_name(get_first(row, "dataset_name", "dataset")),
            normalize_k(get_first(row, "k")),
        ): row
        for row in src_rows
    }

    for row in dst_rows:
        if row.get("design_name") != "MY-Scale":
            continue
        dataset = canonical_dataset_name(row.get("dataset_name"))
        k = normalize_k(row.get("k"))
        src_row = src_by_dataset_and_k.get((dataset, k))
        if src_row is None:
            continue
        row["QPS"] = format_qps(get_first(src_row, "qps", "QPS", default=row["QPS"]))

    write_csv(dst, dst_rows)


def update_qps_vs_recall():
    dst = DATA_DIR / "qps_vs_recall.csv"
    if not dst.exists():
        return

    source_specs = [
        (["SIFT", "sift1M", "sift"], "SIFT"),
        (["GloVe", "glove1_2M", "glove"], "GloVe"),
    ]

    dst_rows = read_csv(dst)

    for src_names, target_dataset in source_specs:
        src = next(
            (
                ROOT / "qps_vs_recall" / f"{src_name}_qps_vs_recall.csv"
                for src_name in src_names
                if (ROOT / "qps_vs_recall" / f"{src_name}_qps_vs_recall.csv").exists()
            ),
            None,
        )
        if src is None:
            continue

        src_rows = read_csv(src)
        new_rows = [
            {
                "dataset_name": target_dataset,
                "design_name": "NasZip",
                "recall": get_first(src_row, "recall", default=""),
                "QPS": format_qps(get_first(src_row, "qps", "QPS", default="")),
            }
            for src_row in src_rows
        ]
        dst_rows = replace_rows(
            dst_rows,
            lambda row, dataset=target_dataset: row.get("dataset_name") == dataset
            and row.get("design_name") == "NasZip",
            new_rows,
        )

    write_csv(dst, dst_rows)


def main():
    update_latency_breakdown()
    update_overall()
    update_overall_scaling()
    update_qps_vs_recall()


if __name__ == "__main__":
    main()
