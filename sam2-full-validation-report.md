# Повна валідація переможця SAM2

Дата: 2026-08-28

## Candidate

- Model: SAM2.1 Hiera-Tiny
- SAM2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Candidate policy: `box_centroid` — tight bounding box + positive centroid point on first frame
- GPU: NVIDIA GeForce RTX 4070 Ti, 12 GB
- PyTorch: `2.13.0+cu130`
- Dataset: DAVIS 2017 TrainVal 480p
- Split: official `val`, 30 sequences
- Predictions: 1999 indexed PNG masks, all validation frames
- Evaluator: official `davisvideochallenge/davis2017-evaluation`, task `semi-supervised`, set `val`

## Official DAVIS results

Source files:

```text
results/davis_full_box_centroid/global_results-val.csv
results/davis_full_box_centroid/per-sequence_results-val.csv
```

| Metric | Value | Direction |
|---|---:|---|
| J&F-Mean | 0.824 | higher is better |
| J-Mean | 0.781 | higher is better |
| J-Recall | 0.854 | higher is better |
| J-Decay | -0.009 | lower decay is better |
| F-Mean | 0.866 | higher is better |
| F-Recall | 0.932 | higher is better |
| F-Decay | -0.007 | lower decay is better |

The unrounded evaluator output was:

```text
J&F-Mean 0.823508
J-Mean   0.780561
J-Recall 0.854205
J-Decay -0.008683
F-Mean   0.866454
F-Recall 0.931725
F-Decay -0.007477
```

## Interpretation

The `box_centroid` policy transfers from the 10-sequence development subset to the complete DAVIS validation split with a strong official `J&F-Mean` of 0.824. The result is a validation score for a semi-supervised prompt setup, not a fully prompt-free segmentation score.

The per-object table shows concentrated failures in difficult sequences/object instances, especially small or rapidly changing objects in `paragliding-launch`, `scooter-black`, `lab-coat` and parts of `kite-surf`. These are useful targets for model-level research, but they must not be used to tune against the held-out data without a new split.

## SA-V validation status

SA-V validation is now complete on the official `sav_val` split.

The user provided an authorized Meta download manifest (`data/sav/download-links.txt`), listing 56 train shards plus separate `sav_val.tar`/`sav_test.tar` archives. Per the deliberate scoping decision recorded in the `sav-download-scope` project memory, only `sav_val.tar` and `sav_test.tar` were fetched and extracted (the 56 train shards were left partially downloaded, not repaired or deleted, to save ~150GB of disk/bandwidth — they are not needed for this validation goal).

`box_centroid` was run on all 155 `sav_val` videos using `scripts/sav_sam2_eval.py`, which mirrors `sam2/tools/vos_inference.py`'s per-object propagation (each of SA-V's manually-annotated objects is prompted and propagated independently, since objects can appear at frames other than 0 and can go invisible). Predictions were written to `results/sav_val_box_centroid/`.

### Official SA-V results

Source files:

```text
results/sav_val_box_centroid_evaluator.log
results/sav_val_box_centroid/results.csv
```

Evaluator: official `sam2/sav_dataset/sav_evaluator.py`, skipping first/last annotated frame per SA-V convention, 155 videos / 293 objects.

| Metric | Value | Direction |
|---|---:|---|
| J&F | 72.1 | higher is better |
| J | 68.6 | higher is better |
| F | 75.6 | higher is better |

### Interpretation

The `box_centroid` policy — selected and frozen on DAVIS alone — transfers positively but incompletely to SA-V: `J&F` drops ~10.2 pp versus the full DAVIS val result (82.4). This is consistent with SA-V being a harder, more diverse benchmark (more objects per video, more occlusion/appearance changes) than DAVIS, not evidence of a broken pipeline: per-object scores in `results/sav_val_box_centroid/results.csv` span from near-perfect (`sav_012436` obj 000, `J&F=98.0`) to near-total collapse (`sav_022456` obj 000, `J&F=0.4`; `sav_043309` obj 000, `J&F=5.0`; `sav_045706` obj 000, `J&F=5.6`). These collapse cases are candidates for future failure-mode analysis, but per the acceptance protocol in `sam2-next-run.md` they must not be used to retune the prompt policy against this held-out split.

## Checkpoint transferability (Hiera-Small)

Candidate: same `box_centroid` policy, same full DAVIS val split (30 sequences, 1999 frames), same official evaluator. Checkpoint: `sam2.1_hiera_small.pt` (official Meta CDN download, SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`), config `configs/sam2.1/sam2.1_hiera_s.yaml`.

| Metric | Hiera-Tiny | Hiera-Small |
|---|---:|---:|
| J&F-Mean | 0.8235 | 0.842 |
| J-Mean | 0.7806 | 0.801 |
| F-Mean | 0.8665 | 0.883 |

The winner does not degrade — it improves slightly — on the larger checkpoint, indicating the `box_centroid` prompt policy is not overfit to Hiera-Tiny specifically. This closes the transferability step (step 8) of the acceptance protocol in `sam2-next-run.md`.

Reproduction:

```bash
.venv/bin/python scripts/davis_sam2_eval.py \
  --davis-root data/davis/DAVIS \
  --sequence-file data/davis/DAVIS/ImageSets/2017/val.txt \
  --checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --config configs/sam2.1/sam2.1_hiera_s.yaml \
  --prompt-policy box_centroid \
  --prediction-root results/davis_full_box_centroid_hiera_small \
  --output results/davis_full_box_centroid_hiera_small.csv

.venv/bin/python davis-evaluation/evaluation_method.py \
  --davis_path data/davis/DAVIS \
  --task semi-supervised \
  --set val \
  --results_path results/davis_full_box_centroid_hiera_small
```

## Acceptance protocol status

Per `sam2-next-run.md`: baseline done, 10 candidates done, full DAVIS held-out val done, SA-V independent validation done, Hiera-Small transferability done. The 3-seed repeat step was explicitly skipped by user decision (2026-08-29) since `box_centroid` is deterministic — the prompt is derived directly from the ground-truth mask with no sampling, so exact repetition would be uninformative; this remains a live requirement for any future stochastic candidate. `sav_test` was deliberately left unopened as the true held-out set.

## Reproduction commands

Generate complete DAVIS predictions:

```bash
.venv/bin/python scripts/davis_sam2_eval.py \
  --davis-root data/davis/DAVIS \
  --sequence-file data/davis/DAVIS/ImageSets/2017/val.txt \
  --prompt-policy box_centroid \
  --prediction-root results/davis_full_box_centroid \
  --output results/davis_full_box_centroid.csv
```

Run official evaluator:

```bash
.venv/bin/python davis-evaluation/evaluation_method.py \
  --davis_path data/davis/DAVIS \
  --task semi-supervised \
  --set val \
  --results_path results/davis_full_box_centroid
```

Generate complete SA-V val predictions:

```bash
.venv/bin/python scripts/sav_sam2_eval.py \
  --sav-root data/sav/sav_val \
  --prompt-policy box_centroid \
  --prediction-root results/sav_val_box_centroid \
  --progress-log results/sav_val_box_centroid_progress.log
```

Run official SA-V evaluator:

```bash
.venv/bin/python sam2/sav_dataset/sav_evaluator.py \
  --gt_root data/sav/sav_val/Annotations_6fps \
  --pred_root results/sav_val_box_centroid \
  -n 8
```

## Quality caveat

The local runner uses a lightweight diagnostic metric during inference and exports official-format indexed PNG masks. The authoritative DAVIS values in this report come from the official evaluator, not the lightweight diagnostic metric.
