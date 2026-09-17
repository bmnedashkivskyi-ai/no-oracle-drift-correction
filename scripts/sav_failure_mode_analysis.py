"""Stratify SA-V val box_centroid J&F scores by ground-truth mask properties.

For each object annotated in SA-V val, computes area-based size bucket, visibility
transitions, and annotated-frame density directly from the ground-truth PNG masks
(SA-V val/test ship no metadata JSON, unlike SA-V train), then joins these against
the per-object J&F scores from the official sav_evaluator.py output to find which
properties predict box_centroid failure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Same thresholds as SA-V train's masklet_size_bucket (pixel area).
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96


def size_bucket(mean_area: float) -> str:
    if mean_area < SMALL_MAX:
        return "small"
    if mean_area < MEDIUM_MAX:
        return "medium"
    return "large"


def load_results(results_csv: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows = {}
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["sequence    "].strip()
            if seq == "Global score":
                continue
            obj = row["obj"].strip()
            rows[(seq, obj)] = {
                "jf": float(row["  J&F"]),
                "j": float(row["    J"]),
                "f": float(row["    F"]),
            }
    return rows


def analyze_object(obj_dir: Path, pred_dir: Path) -> dict:
    frame_files = sorted(obj_dir.glob("*.png"))
    areas = []
    visible_flags = []
    for f in frame_files:
        arr = np.array(Image.open(f))
        area = int((arr > 0).sum())
        areas.append(area)
        visible_flags.append(area > 0)

    visibility_changes = sum(
        1 for i in range(1, len(visible_flags)) if visible_flags[i] != visible_flags[i - 1]
    )
    mean_area = float(np.mean(areas)) if areas else 0.0

    # Fill ratio and prompt-frame IoU are computed at the first annotated frame,
    # i.e. the frame box_centroid actually derives its box/point prompt from.
    first = frame_files[0]
    gt0 = np.array(Image.open(first)) > 0
    ys, xs = np.nonzero(gt0)
    box_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1) if len(xs) else 0
    fill_ratio = (gt0.sum() / box_area) if box_area else 0.0

    prompt_frame_iou = None
    pred0_path = pred_dir / first.name
    if pred0_path.exists():
        pred0 = np.array(Image.open(pred0_path)) > 0
        inter = (gt0 & pred0).sum()
        union = (gt0 | pred0).sum()
        prompt_frame_iou = float(inter / union) if union else 0.0

    return {
        "num_annotated_frames": len(frame_files),
        "mean_mask_area_px": round(mean_area, 1),
        "size_bucket": size_bucket(mean_area),
        "visibility_changes": visibility_changes,
        "ever_invisible": not all(visible_flags),
        "box_fill_ratio": round(fill_ratio, 4),
        "prompt_frame_iou": round(prompt_frame_iou, 4) if prompt_frame_iou is not None else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, default=Path("data/sav/sav_val/Annotations_6fps"))
    parser.add_argument("--pred-root", type=Path, default=Path("results/sav_val_box_centroid"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/sav_val_box_centroid/results.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    args = parser.parse_args()

    scores = load_results(args.results_csv)
    fieldnames = [
        "video", "object", "num_objects_in_video", "jf", "j", "f",
        "num_annotated_frames", "mean_mask_area_px", "size_bucket",
        "visibility_changes", "ever_invisible", "box_fill_ratio", "prompt_frame_iou",
    ]
    out_rows = []
    for video_dir in sorted(args.gt_root.iterdir()):
        if not video_dir.is_dir():
            continue
        object_dirs = sorted(d for d in video_dir.iterdir() if d.is_dir())
        num_objects = len(object_dirs)
        for obj_dir in object_dirs:
            key = (video_dir.name, obj_dir.name)
            if key not in scores:
                continue
            pred_dir = args.pred_root / video_dir.name / obj_dir.name
            stats = analyze_object(obj_dir, pred_dir)
            out_rows.append({
                "video": video_dir.name,
                "object": obj_dir.name,
                "num_objects_in_video": num_objects,
                **scores[key],
                **stats,
            })

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} object rows to {args.output_csv}")

    # Quick stratified summary printed to stdout.
    def summarize(rows, key_fn, label):
        groups: dict = {}
        for r in rows:
            groups.setdefault(key_fn(r), []).append(r["jf"])
        print(f"\n-- {label} --")
        for k, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
            print(f"{k!s:>12}  n={len(vals):3d}  mean J&F={np.mean(vals):5.1f}  median={np.median(vals):5.1f}")

    summarize(out_rows, lambda r: r["size_bucket"], "By size bucket")
    summarize(out_rows, lambda r: r["ever_invisible"], "By ever_invisible")
    summarize(
        out_rows,
        lambda r: (
            "1" if r["num_objects_in_video"] == 1 else
            "2-3" if r["num_objects_in_video"] <= 3 else "4+"
        ),
        "By num_objects_in_video",
    )
    summarize(
        out_rows,
        lambda r: (
            "0" if r["visibility_changes"] == 0 else
            "1-2" if r["visibility_changes"] <= 2 else "3+"
        ),
        "By visibility_changes",
    )

    def fill_bucket(v):
        if v == "":
            return "n/a"
        if v < 0.2:
            return "<0.2 (sparse)"
        if v < 0.5:
            return "0.2-0.5"
        return ">=0.5 (compact)"

    summarize(out_rows, lambda r: fill_bucket(r["box_fill_ratio"]), "By box_fill_ratio (gt_area / box_area at prompt frame)")

    def iou_bucket(v):
        if v == "":
            return "n/a"
        if v < 0.3:
            return "<0.3 (prompt fails)"
        return ">=0.3 (prompt ok)"

    summarize(out_rows, lambda r: iou_bucket(r["prompt_frame_iou"]), "By prompt_frame_iou")

    # Split failures (J&F < 40) into prompt-frame failures vs propagation-drift failures.
    failures = [r for r in out_rows if r["jf"] < 40]
    prompt_failures = [r for r in failures if r["prompt_frame_iou"] != "" and r["prompt_frame_iou"] < 0.3]
    drift_failures = [r for r in failures if r["prompt_frame_iou"] != "" and r["prompt_frame_iou"] >= 0.3]
    print(f"\n-- Failure decomposition (J&F < 40, n={len(failures)}) --")
    print(f"prompt-frame failures (prompt_frame_iou < 0.3): {len(prompt_failures)}")
    print(f"propagation-drift failures (prompt_frame_iou >= 0.3 but video J&F < 40): {len(drift_failures)}")

    worst = sorted(out_rows, key=lambda r: r["jf"])[:15]
    print("\n-- 15 worst objects --")
    for r in worst:
        print(
            f"{r['video']} obj{r['object']}  J&F={r['jf']:5.1f}  "
            f"size={r['size_bucket']:>6}  area={r['mean_mask_area_px']:7.0f}px  "
            f"fill_ratio={r['box_fill_ratio']}  prompt_iou={r['prompt_frame_iou']}  "
            f"n_objs={r['num_objects_in_video']}  vis_changes={r['visibility_changes']}  "
            f"ever_invisible={r['ever_invisible']}"
        )


if __name__ == "__main__":
    main()
