"""E1: object-level risk features from an UNCORRECTED pass (sam2-drift-recovery-
next-directions.md SS7, direction E).

D1-D4 all tune or combine the A2 circuit breaker, which can only act AFTER
n_corrections>=3 real corrections have already been made -- if the first 1-3
corrections do the damage (sav_004755, sav_017171), the breaker is structurally
too late. Real pre-selection needs to decide whether to run P5 correction on an
object AT ALL, before the first correction, using signals from a pass that
never corrects.

This script propagates each object with the SAME area-ratio / position-jump
detection logic as sav_motion_prior_reprompt_eval.py, but never acts on a
trigger: it just counts triggers and volatility, and keeps propagating from
the model's own (uncorrected) next-frame mask. Cheaper than a corrected pass
(no add_new_points_or_box re-prompt calls, no O(N^2) re-entry into
propagate_in_video -- a single continuous call per object) and produces
per-object features usable to retroactively check, against already-known
capstone per-object J&F deltas, whether "this object would need correction"
is predictable before ever attempting one.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_motion_prior_reprompt_eval import mask_centroid  # noqa: E402
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def evaluate_object_dry_run(
    predictor,
    state,
    object_id: int,
    per_frame,
    prompt_policy: str,
    ema_alpha: float,
    low_ratio: float,
    high_ratio: float,
    min_area_eps: float,
    jump_ratio: float,
) -> dict:
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    last_mask = mask_logits[0, 0] > 0
    ema_area = float(last_mask.sum().item())
    last_centroid = mask_centroid(last_mask)
    prev_point = (first_frame_idx, last_centroid)
    velocity = (0.0, 0.0)

    n_frames = 0
    n_area_triggers = 0
    n_position_triggers = 0
    area_ratio_devs: list[float] = []
    position_jump_scaled: list[float] = []

    # E1 timing variant (sam2-drift-recovery-next-directions.md SS8, "not yet
    # explored" note): the raw rate/amplitude features tested first showed no
    # signal (R^2=0.026) -- they can't distinguish "trigger correctly caught a
    # real identity switch" from "trigger misfired on legitimate motion". The
    # two documented capstone catastrophes (sav_004755, sav_017171) share a
    # different, more specific shape: the FIRST correction attempt itself was
    # fatal. That's a claim about WHEN/HOW STRONG the first trigger is, not
    # about how often triggers happen over the whole video -- a signal the
    # rate-based features average away. Track it directly instead of
    # inferring it post-hoc: frame position of the first trigger (as a
    # fraction of the object's elapsed frames), its severity relative to
    # threshold (how far past the line it fired, not just that it fired), and
    # what fraction of all triggers land in the first 10%/20% of the video.
    first_trigger_rel_frame: int | None = None
    first_trigger_reason: str | None = None
    first_trigger_severity: float | None = None
    trigger_rel_frames: list[int] = []

    for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
        state, start_frame_idx=first_frame_idx + 1
    ):
        mask = mask_logits[0, 0] > 0
        area = float(mask.sum().item())
        n_frames += 1
        rel_frame = frame_idx - first_frame_idx

        is_area_drift = ema_area >= min_area_eps and (
            area == 0 or area / ema_area < low_ratio or area / ema_area > high_ratio
        )
        area_ratio_dev = None
        if ema_area >= min_area_eps and area > 0:
            area_ratio_dev = abs(area / ema_area - 1.0)
            area_ratio_devs.append(area_ratio_dev)
        if is_area_drift:
            n_area_triggers += 1
            trigger_rel_frames.append(rel_frame)
            # severity = how far past the nearer threshold edge, in units of
            # the gap between 1.0 (no deviation) and that edge -- >0 means
            # past the line, larger means further past it. Vanish (area==0)
            # is maximal severity by construction.
            if area == 0:
                severity = float("inf")
            else:
                ratio = area / ema_area
                severity = (low_ratio - ratio) / low_ratio if ratio < low_ratio else (ratio - high_ratio) / high_ratio
            if first_trigger_rel_frame is None:
                first_trigger_rel_frame, first_trigger_reason, first_trigger_severity = rel_frame, "area", severity

        centroid = mask_centroid(mask)
        if centroid is not None and last_centroid is not None:
            dt = frame_idx - prev_point[0]
            predicted = (
                last_centroid[0] + velocity[0] * dt,
                last_centroid[1] + velocity[1] * dt,
            )
            jump = ((centroid[0] - predicted[0]) ** 2 + (centroid[1] - predicted[1]) ** 2) ** 0.5
            scale = max((ema_area ** 0.5), 1.0)
            position_jump_scaled.append(jump / scale)
            if jump > jump_ratio * scale:
                n_position_triggers += 1
                trigger_rel_frames.append(rel_frame)
                severity = (jump / scale) / jump_ratio - 1.0
                if first_trigger_rel_frame is None:
                    first_trigger_rel_frame, first_trigger_reason, first_trigger_severity = rel_frame, "position", severity

        # Always accept -- never correct -- update trajectory from the
        # model's own mask exactly like the "accepted frame" branch of the
        # corrected evaluator, just unconditionally.
        if centroid is not None and last_centroid is not None:
            dt = max(frame_idx - prev_point[0], 1)
            velocity = ((centroid[0] - last_centroid[0]) / dt, (centroid[1] - last_centroid[1]) / dt)
            prev_point = (frame_idx, centroid)
        last_centroid = centroid if centroid is not None else last_centroid
        last_mask = mask
        ema_area = ema_alpha * area + (1 - ema_alpha) * ema_area

    n_triggers = n_area_triggers + n_position_triggers
    n_early_10 = sum(1 for r in trigger_rel_frames if r <= 0.10 * n_frames)
    n_early_20 = sum(1 for r in trigger_rel_frames if r <= 0.20 * n_frames)
    return {
        "n_frames": n_frames,
        "n_area_triggers": n_area_triggers,
        "n_position_triggers": n_position_triggers,
        "n_triggers": n_triggers,
        "trigger_rate": n_triggers / max(n_frames, 1),
        "area_dev_mean": statistics.fmean(area_ratio_devs) if area_ratio_devs else 0.0,
        "area_dev_max": max(area_ratio_devs) if area_ratio_devs else 0.0,
        "pos_jump_mean": statistics.fmean(position_jump_scaled) if position_jump_scaled else 0.0,
        "pos_jump_max": max(position_jump_scaled) if position_jump_scaled else 0.0,
        "first_trigger_frac": (first_trigger_rel_frame / n_frames) if first_trigger_rel_frame is not None and n_frames > 0 else -1.0,
        "first_trigger_reason": first_trigger_reason or "",
        "first_trigger_severity": first_trigger_severity if first_trigger_severity is not None else -1.0,
        "frac_triggers_first_10pct": (n_early_10 / n_triggers) if n_triggers > 0 else 0.0,
        "frac_triggers_first_20pct": (n_early_20 / n_triggers) if n_triggers > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--prompt-policy", default="largest_component_box_centroid")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--low-ratio", type=float, default=0.35)
    parser.add_argument("--high-ratio", type=float, default=2.5)
    parser.add_argument("--min-area-eps", type=float, default=50.0)
    parser.add_argument("--jump-ratio", type=float, default=8.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    video_list_path = args.video_list or (args.sav_root / f"{args.sav_root.name}.txt")
    video_names = [line.strip() for line in video_list_path.read_text().splitlines() if line.strip()]
    if args.max_videos:
        video_names = video_names[: args.max_videos]

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    fieldnames = [
        "video", "object_id", "n_frames", "n_area_triggers", "n_position_triggers",
        "n_triggers", "trigger_rate", "area_dev_mean", "area_dev_max",
        "pos_jump_mean", "pos_jump_max",
        "first_trigger_frac", "first_trigger_reason", "first_trigger_severity",
        "frac_triggers_first_10pct", "frac_triggers_first_20pct",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    log_lines = []

    for i, video_name in enumerate(video_names):
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        if not video_dir.exists() or not ann_dir.exists():
            continue
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)
        if not object_inputs:
            continue

        t0 = time.perf_counter()
        state = predictor.init_state(str(video_dir))
        for object_id, per_frame in sorted(object_inputs.items()):
            feats = evaluate_object_dry_run(
                predictor, state, object_id, per_frame, args.prompt_policy,
                args.ema_alpha, args.low_ratio, args.high_ratio,
                args.min_area_eps, args.jump_ratio,
            )
            rows.append({"video": video_name, "object_id": f"{object_id:03d}", **feats})
            line = (
                f"[{i + 1}/{len(video_names)}] video={video_name} obj={object_id:03d} "
                f"n_frames={feats['n_frames']} n_triggers={feats['n_triggers']} "
                f"trigger_rate={feats['trigger_rate']:.4f}"
            )
            print(line, flush=True)
            log_lines.append(line)
            if args.progress_log:
                args.progress_log.write_text("\n".join(log_lines) + "\n")
            with args.output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote {len(rows)} object risk-feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()
