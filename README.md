# TechTronics — IEEE SP Cup 2027 (FaceVitals)

2027 SP Cup, Team TechTronics, Department of Electronic and Telecommunication Engineering, University of Moratuwa, Sri Lanka.

**Challenge:** contactless estimation of Heart Rate, SpO₂ and Blood Pressure from RGB facial video (remote photoplethysmography, rPPG).
**Round 1:** Heart Rate only. Deadline **30 November 2026**.

---

## Pipeline

```
Video ─► (Constraint simulation) ─► Face detection & tracking ─► ROI + skin mask ─► RGB traces ─► rPPG signal ─► HR ─► Metrics
```

## Requirements

- MATLAB **R2025a or newer** (developed on R2026b).
- Toolboxes:
  - Computer Vision Toolbox
  - Image Processing Toolbox
  - Deep Learning Toolbox
  - Signal Processing Toolbox
- Add-on: **Computer Vision Toolbox Model for RetinaFace Face Detection**. Install it from Home → Add-Ons; it provides `faceDetector`.

Run everything from the repo root after adding the code to the path:
```matlab
addpath(genpath("src"))
```

## Face module (video -> ROI traces)

Tracks one target face (also with bystanders, C3), works at VGA and QVGA (C4),
and flags regions covered by a mask (C7). Output: per-frame mean RGB of each skin region.

### Run
From the repo root in MATLAB:
```matlab
addpath(genpath("src"))

% one video -> one trace file
out = process_video("path/to/video.avi", Subject="subject1", Condition="Baseline", Constraint="None");
export_traces(out);                       % -> outputs/traces/UBFC2_subject1_Baseline_None.mat

% full benchmark: test clips + traces + tables + debug videos
T = run_pipeline(Quick=true);             % first 10 s per video
T = run_pipeline();                       % full videos, subjects [1 3 4 12 17 26 31 34]
```

| Condition | Constraint | Test clip |
|---|---|---|
| `Baseline` | None | original UBFC video (640×480) |
| `LowResolutionQVGA` | C4 | bicubic resize to 320×240 |
| `MultiFace` | C3 | second real subject composited beside the target |
| `Masked` | C7 | synthetic surgical mask over the lower face |
| `Stress` | C3C4C7 | MultiFace + Masked + QVGA |

### Configuration: `config/face_config.json`
Edit this file instead of the code. Any field can be left out, and the
defaults from `src/face/face_params.m` are used. To use another file, pass
`process_video(..., Config="my.json")` or `run_pipeline(Config="my.json")`.

| Key | Meaning |
|---|---|
| `roi.names` | **regions in use**, e.g. `["forehead","lcheek","rcheek"]` (default) |
| `roi.<name>` | region box as face-box fractions `[u1 v1 u2 v2]` (u: left→right, v: top→bottom). Also predefined but unused: `glabella`, `nose`, `lmalar`, `rmalar` |
| `det.every` | run the face detector every N frames (KLT tracks in between) |
| `det.threshold` | face detection confidence |
| `det.device` | `"cpu"` (default; bit-reproducible results) or `"gpu"` / `"auto"` (faster, but small run-to-run differences) |
| `trk.targetPolicy`, `trk.targetBox` | how the target face is chosen (largest + central, or a manual box) |
| `skin.cb`, `skin.cr`, `skin.yMin`, `skin.satMax` | YCbCr skin-pixel thresholds |
| `occl.onBelow` / `occl.offAbove` | skin-ratio thresholds for the occlusion flag |
| `occl.reference`, `occl.relOn` | region the other regions are compared with, and the relative threshold |
| `smooth.enable`, `smooth.minCutoff`, `smooth.beta` | ROI smoothing (One-Euro filter) |
| `clip.*` | test-clip generation (bystander size, mask shape and colour) |

To add a region, define it under `roi` and list it in `roi.names`. For example,
`"names": ["forehead","lcheek","rcheek","glabella"]` gives an extra
`rgb_glabella` output and one more column in `npix` / `occluded`.

### Output (input for signal processing)
There is one `.mat` file per video, saved with `-v7`, readable by MATLAB `load` and
Python `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`.
N is the number of frames (**no frame is dropped**), and R is the number of regions.

| Variable | Shape | Meaning |
|---|---|---|
| `t` | N×1 | frame time in seconds, **read from the video** (not k/fps) |
| `rgb_<region>` | N×3 | mean R,G,B (0–255) of the skin pixels in that region, target face only (cheeks: viewer's side) |
| `rgb_face` | N×3 | pixel-weighted mean over the **non-occluded** regions (the recommended single trace) |
| `roi_names` | 1×R | region names, which give the column order of `npix`, `skin_ratio`, `occluded` |
| `npix`, `skin_ratio` | N×R | skin pixels used, and the skin share of each region |
| `occluded` | N×R | 1 = region occluded (e.g. mask): down-weight or drop it |
| `quality` | N×1 | 0 = no target (RGB is NaN), 1 = tracked, 2 = confirmed by the detector |
| `n_faces`, `target_id` | N×1 | faces visible, and the track ID of the target |
| `roi_data` | N×1 struct | `face_box` [x y w h], one 4×2 polygon per region, `all_boxes`, `track_ids` |
| `meta` | struct | source, condition, constraint, resolution, fps, timing, and the `params` used |

Rows with `quality = 0` are NaN. Interpolate across them or mask them out before filtering.

### Benchmark
`run_pipeline` / `run_benchmark` write these to `outputs/`:
- `tables/face_benchmark.csv` (per clip) and `face_benchmark_summary.csv` (per condition): target found %, ID switches, face count, ROI jitter, skin pixels, occlusion accuracy, ms/frame, and a baseline HR (POS) compared against the contact PPG (MAE, RMSE, r, SUC@3/5);
- `tables/face_bland_altman.csv` and `figures/bland_altman_by_condition.png`;
- `figures/*_debug.avi`: overlay videos (target / bystander boxes, ROIs, occlusion flags).

## Repository layout

```
├── config/                 # shared parameter files (single source of truth)
├── data/                   # gitignored: datasets and generated videos
│   ├── raw/                #   original datasets (UBFC-rPPG, organizer data, ...)
│   └── degraded/           #   videos with simulated constraints
├── docs/
│   └── competition/        # official competition documents
├── src/
│   ├── degrade/            # constraint simulation (resolution, frame rate, bitrate, ...)
│   ├── face/               # face detection, tracking, ROI extraction, skin mask
│   ├── dataio/             # video / ground-truth loaders, data export
│   ├── signal/             # rPPG signal processing and HR estimation
│   ├── eval/               # metrics: MAE, RMSE, Pearson r, Bland–Altman
│   └── tools/              # visual debugging, timing, utilities
├── deploy/
│   └── pi5/                # Raspberry Pi 5 deployment
├── tests/
├── outputs/                # gitignored: traces, ROI data, figures, tables
├── report/                 # technical report (IEEE conference format)
│   └── figures/
└── submission/             # official submission package
    └── Round1_Baseline/
```

## Data

Datasets live in `data/raw/` (gitignored; never commit videos or `.mat` data):
```
data/raw/UBFC/DATASET_2/subjectN/vid.avi
data/raw/UBFC/DATASET_2/subjectN/ground_truth.txt
```
- UBFC subjects used: 1, 3, 4, 12, 17, 26, 31, 34. Excluded: 27 (no image consent: never show its face) and 11, 18, 20, 24 (pulse-oximeter dropouts).
- Organizer dataset: covered by a data agreement. Keep it on team machines only.
- UBFC citation: S. Bobbia et al., "Unsupervised skin tissue segmentation for remote photoplethysmography", *Pattern Recognition Letters*, 2019.
