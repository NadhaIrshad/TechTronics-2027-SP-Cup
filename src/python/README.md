# Python pipeline: C2 skin-tone evaluation

Runs the **same HR pipeline on every dataset** and reports the error per skin-tone group (ITA).
The face module is a line-by-line port of the MATLAB one (`src/face`), so it runs without
MATLAB R2025a+. HR methods compared (`--method`):

| Method | Steps |
|---|---|
| `pos` | baseline: skin pixels of all regions averaged → POS → spectral peak (`sanity_hr.m`) |
| `pos_bg` | `pos` with background-reference regression (lighting flicker) |
| `pos_kf` | Kalman denoising of the RGB trace before POS (team method, `kalman_denoise.m`) |
| `pos_eq` | POS per region, spectra fused with equal weights (control) |
| `pos_snr` | POS per region, spectra fused with per-window SNR weights |
| `pos_snr_bg` | `pos_snr` with background regression per region |
| `pos_kf_snr` | Kalman per region → POS per region → SNR fusion |
| **`pos_kf_snr_bg`** | **`pos_kf_snr` with background regression: the full method** |

## Requirements

Python 3.11 with `opencv-python`, `torch` + `torchvision` (CPU is enough), `mediapipe`,
`scikit-image`, `scipy`, `pandas`, `openpyxl`.

RetinaFace weights (`mobilenet0.25_Final.pth`, 1.8 MB, WIDER FACE) are downloaded on first use to
`models/retinaface/` (gitignored), checked against a fixed SHA-256 and loaded with
`torch.load(weights_only=True)`.

Datasets are read from the folder that contains the repo (`D:/SPCUP`), as downloaded:
`UBFC-rPPG/`, `MPU rppg/`, `CLBP-400_Samples/`, `rppg_biases-main/`, `SRMC_N/`.
Change it with `--data-root` or the `FACEVITALS_DATA` environment variable.
UBFC subjects in `data/raw/UBFC/DATASET_2/` are also picked up.

## Run

```bash
# one dataset at a time (one process; safest on a laptop with ~12 GB RAM)
python -u src/python/run_c2.py --datasets AiPEX       # ~12 min, traces cached afterwards
python -u src/python/run_c2.py --datasets UBFC        # ~10 min with the tracker
python -u src/python/run_c2.py --datasets CLBP        # 4K video, ~1 h with the tracker

# merge all dataset tables and report
python src/python/run_c2.py --analyze-only --method compare        # all methods side by side
python src/python/run_c2.py --analyze-only --method pos_kf_snr_bg  # C2 tables for one method

# parallel runs: independent processes, one per shard, then --analyze-only
python src/python/run_c2.py --shard 1/4      # ... up to --shard 4/4

# quick test
python src/python/run_c2.py --datasets UBFC --subjects subject1 --max-frames 600

# skin-tone survey only (fast, includes SRMC-N, which has no ground truth), with thumbnails
python src/python/ita_survey.py
```

MediaPipe on Windows fails when one process spawns several workers that load it at once, so
parallel runs use independent processes (`--shard`), not a process pool. Traces are cached in
`outputs/traces_py/`; `--rerun` ignores the cache.

## What it does

### Face module (`--face-module tracker`, default)

A line-by-line port of the MATLAB face module (`src/face/`), same algorithm and the same
parameters (`face_params.m` defaults + `config/face_config.json`):

| MATLAB | Python | Note |
|---|---|---|
| `faceDetector("small-network")` (RetinaFace) | `facevitals/retinaface.py`: RetinaFace MobileNet-0.25 | Same detector and backbone, public WIDER FACE weights (MathWorks' trained model cannot be loaded in Python, so boxes can differ slightly). Threshold 0.5, MinSize 10 px, CPU |
| `FaceTracker.m` | `facevitals/tracker.py` `FaceTracker` | Detection every `det.every` frames (`everyLost` when lost), search window after KLT failure, target choice (largest-central / `trk.targetBox`), re-acquisition rules, re-anchoring at IoU < 0.75 |
| `configureKalmanFilter("ConstantVelocity", …)` | `KalmanCV` | State [x vx y vy], P0 = diag(200, 50), Q = diag(100, 25), R = 100 |
| `assignDetectionsToTracks` | `assign_detections_to_tracks` | Hungarian on 1 − IoU, padded with cost of non-assignment 0.5, gate at IoU < 0.1 |
| `detectMinEigenFeatures` + `vision.PointTracker` | `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK` | Min-eigenvalue corners (300, quality 0.01, inset box), 31×31 window, 3 levels, forward-backward error ≤ 2 px |
| `estgeotform2d("similarity")` | `cv2.estimateAffinePartial2D` (RANSAC, 3 px) | Scale sanity check 0.5–2, point refresh below 50 % |
| `OneEuroFilter.m`, `roi_polygons.m`, `apply_tform.m`, `box_iou.m`, `clip_box.m` | same names in `tracker.py` | Coordinates 0-based |
| `process_video.m`, `roi_skin_stats.m`, `skin_mask.m` | `face.tracked_face_traces`, `face.roi_skin_stats`, `face.skin_mask` | One-Euro on all polygon corners + box; YCbCr skin mask with MATLAB `rgb2ycbcr` coefficients; occlusion = absolute AND relative rule with hysteresis on the EMA skin ratio, or < 10 skin px; NaN frames when lost |
| `sanity_hr.m`, `ppg_hr.m`, `spectral_hr.m` | `facevitals/rppg.py` | 30 Hz resampling on real timestamps, POS, Butterworth 0.7–3 Hz, 10 s windows every 1 s, periodogram peak |

Check on UBFC subject1 (full video): MATLAB README 1.47 bpm, this port 1.41 bpm; the detector ran
on 155 of 1547 frames (every 10th), one face, no false occlusion. Speed ≈ 30 ms/frame at 640×480
on CPU (plus a one-time ~10 s torch import).

Extra outputs for the C2 evaluation (not in MATLAB): the background trace (`rgb_bg`, below) and
ITA samples: `rgb2lab` (sRGB, D65) on the same skin pixels as HR, every 30th frame; median L\*, b\*
per region; a region with L\* > 15 below the median is dropped as hair/shadow.

`--face-module simple` keeps the earlier development version (MediaPipe largest face, box
extended 15 % upwards, exponential smoothing, absolute occlusion rule only). It has its own trace
cache (`outputs/traces_py/` vs `outputs/traces_py/tracker/`) and its tables carry a `_simple`
suffix, so the two are never merged. AiPEX clips are crops, so face detection is skipped and both
modules give identical AiPEX results.

### Background-reference rejection (`--method pos_bg`)

Light flicker, camera exposure oscillation and coloured lighting (e.g. an RGB keyboard) modulate
the whole scene. When the modulation has a colour component, POS does not remove it and its
frequency can win the spectral peak (CLBP Subject068: a constant 2.0 Hz = 120 bpm rhythm in face,
wall and hijab; MPU: ~55–59 bpm keyboard lighting in face, shirt and curtain).

1. **Background trace** (`face.background_rgb`): per frame, mean RGB of everything outside the face
   box enlarged 1.6×, with skin pixels (hands, neck: they carry a pulse) and clipped pixels removed,
   computed on a ¼-size copy of the frame. Saved as `rgb_bg` with `bg_npix` and `face_box`.
2. **Rejection** (`rppg.sanity_hr_bg`, `mode="regress"`): face and background go through the same
   resampling, POS and band-pass. In each 10 s window the face pulse signal is least-squares fitted
   by the background signals (background POS + its 3 normalised, band-passed channels) and the fit
   is subtracted before the periodogram. The background has no pulse, so only shared illumination
   is removed.

`mode="spectral"` (divide the face spectrum by the background spectrum) is kept for comparison:
10 s windows make peaks ±10 bpm wide and the flanks of a strong flicker peak survive.
Every subject row has both results: `mae`, `suc5_pct`, ... (baseline) and `bg_mae`, `bg_suc5_pct`, ...

```bash
python src/python/run_c2.py --analyze-only --method compare   # baseline vs background reference
python src/python/run_c2.py --analyze-only --method pos_bg    # skin-tone statistics with the fix
```

AiPEX clips are crops without background, so `pos_bg` equals the baseline there.

### SNR-weighted region fusion (`--method pos_snr`, `pos_snr_bg`)

The baseline averages the regions' pixels (`rgb_face`), so a large noisy region outweighs a small
clean one. `rppg.sanity_hr_snr` instead runs POS on every region separately and, in each 10 s window:

1. computes each region's periodogram and its SNR (de Haan & Jeanne 2013): power within ±0.1 Hz of
   the in-band peak and ±0.2 Hz of its first harmonic, divided by the rest of the 0.7–3 Hz band;
2. normalises each spectrum to unit in-band power and weights it by SNR^`snr_power` (1.0);
3. sums the weighted spectra; HR = peak of the sum.

Weights are recomputed per window, so a region that goes bad for a few seconds (shadow, hand,
mask edge) only loses weight for those seconds. `pos_snr_bg` first regresses the background signals
out of every region (as in `pos_bg`); it equals `pos_snr` on AiPEX. Rows contain `snr_mae`,
`snrbg_mae`, ... and `snr_w_<region>`: the mean share of each region in the fused spectrum.

**Control `pos_eq`:** the same per-region POS and spectrum fusion with equal weights (SNR⁰ = 1).
`pos_eq` vs `pos` measures the gain from fusing region *spectra* instead of averaging *pixels*;
`pos_snr` vs `pos_eq` measures what the SNR weights add on top. Columns: `eq_mae`, `eq_suc5_pct`, ...

### Kalman denoising (`pos_kf`) and the combination (`pos_kf_snr`, `pos_kf_snr_bg`)

`rppg.kalman_denoise` is a port of `src/signal/kalman_denoise.m` from the team branch
`thamilezai_dev` (checked against the MATLAB function on a real AiPEX trace: max difference
3e-14, same NaN handling). Per RGB channel, a random-walk level is tracked with measurement noise
R0·median(npix)/npix(k) (fewer skin pixels → trusted less); the steady-state filter is a
first-order low-pass with −3 dB at 4 Hz, above the HR band. It works **before POS** (cleaner
input, so POS's σ₁/σ₂ weighting is not dominated by frame noise); SNR fusion works **after POS**
(chooses between regions), so they can be stacked:

- `pos_kf`: denoise `rgb_face` (weighted by its non-occluded skin pixels), then baseline POS.
- `pos_kf_snr`: denoise each region (weighted by its own skin pixels), then per-region POS + SNR fusion.
- `pos_kf_snr_bg`: as `pos_kf_snr` with the background regression.

AiPEX crops carry no pixel counts, so there the filter is unweighted (a fixed exponential
smoother). Recording-level datasets also store `*_mae_meanagg`: the MAE when the recording HR is
the **mean** of the window HRs, as in `aipex_hr.m`, next to our **median**.

Limitation: SNR is measured around each region's own peak, so a region locked onto a strong
artefact can look "clean". The background regression in `pos_snr_bg` reduces that risk.

Reference HR:
- **UBFC:** from the contact PPG, same windows and method as the estimate. Subject MAE = mean over its windows.
- **MPU:** the oximeter's HR column, averaged inside each 10 s window. HR computed from the exported
  PPG column disagreed with the device HR by 11–45 bpm on recordings 8, 9 and 10, so it is not used.
- **AiPEX, CLBP:** one value per recording (AiPEX: Masimo mean; CLBP: spot reading). Estimate = median of the window HRs.
- **SRMC-N:** none. ITA only.

AiPEX clips are pre-cropped regions (no full face), so the skin mask is applied to the whole crop
and face detection is skipped. That is a different path from the other datasets, which is why the
regression below includes a dataset term.

## Statistics (`config/c2_config.json`)

- **Groups:** six ITA classes (Very light > 55°, Light 41–55°, Intermediate 28–41°, Tan 10–28°,
  Brown −30–10°, Dark < −30°, roughly Fitzpatrick I–VI) and three merged groups (I–II, III–IV, V–VI)
  because some classes have very few subjects.
- **Acceptance:** a group passes if |group MAE − population MAE| ≤ `tau_bpm` (2.0 bpm). τ was fixed
  before any result was computed.
- **95 % CI:** bootstrap over subjects.
- **Regression:** subject MAE ~ ITA + dataset, so a harder dataset is not mistaken for a skin-tone effect.
- **Within-dataset:** Spearman correlation between ITA and MAE in each dataset with ≥ 5 subjects.

Outputs in `outputs/tables/`: `c2_subjects.csv`, `c2_windows.csv`, `c2_groups.csv`,
`c2_regression.csv`, `c2_within_dataset.csv`.

## Results: AiPEX rppg_biases (C2)

88 recordings (India 44, Sierra Leone 44). Each recording is three pre-cropped ROI clips
(forehead 60×30, cheeks 25×25 px, upscaled to 200×200), 15 fps, compressed. Reference: one
Masimo mean HR per recording; estimate = **median** of the 10 s window HRs (the *mean* column
reproduces `aipex_hr.m` on `thamilezai_dev`). AiPEX has no background and no face detection, so
`*_bg` = the same method without background, and both face modules give identical results.

### Overall

| Method | MAE (median of windows) | MAE (mean of windows) | Median MAE | Within 5 bpm | Bias |
|---|---|---|---|---|---|
| `pos` baseline (pixels averaged) | 16.01 | 18.56 | 16.50 | 13.6 % | +15.8 |
| `pos_eq` region spectra, equal weights | 14.06 | 16.49 | 13.30 | 19.3 % | – |
| `pos_kf` Kalman denoising (team) | 9.90 | 13.11 | 9.24 | 25.0 % | +9.0 |
| `pos_snr` region spectra, SNR weights | 8.20 | 11.80 | 5.77 | 43.2 % | +5.7 |
| **`pos_kf_snr` Kalman + SNR fusion** | **5.76** | **7.45** | **4.80** | **53.4 %** | **+0.9** |

- **Reproduces the team MATLAB result:** with the mean of windows, `pos` 18.56 and `pos_kf` 13.11
  match `aipex_hr.m` (18.6 → 13.1).
- **The combination beats each part** (> 1 bpm, Wilcoxon): vs `pos_kf` 62 better / 19 worse,
  vs `pos_snr` 56 / 25, vs `pos` 72 / 13; all p < 0.001.
- **Why they stack:** Kalman removes fast frame noise *before* POS (POS's σ₁/σ₂ weighting is not
  dominated by noise); SNR fusion picks the clearest region *after* POS, per window.
- `pos_snr` vs `pos_eq`: 69 better / 15 worse, p < 0.001. The fusion gain comes from the **SNR
  weights**, not only from fusing spectra (weights average ≈ ⅓ per region over a recording but
  differ per window).
- AiPEX paper (Dasari et al. 2021) on the same videos: POS 14.0, CHROM 11.6, BKF 5.8 MAE
  (48 % within 5 bpm).

### By country (MAE, median of windows)

| Country | n | `pos` | `pos_eq` | `pos_kf` | `pos_snr` | **`pos_kf_snr`** | `pos_kf_snr` median / within 5 / bias |
|---|---|---|---|---|---|---|---|
| India | 44 | 15.76 | 12.89 | 9.65 | 7.87 | **6.17** | 4.80 / 54.5 % / +0.4 |
| Sierra Leone | 44 | 16.25 | 15.22 | 10.14 | 8.54 | **5.35** | 4.74 / 52.3 % / +1.3 |

### By skin tone (ITA group), with the C2 acceptance test

Population MAE over the 85 recordings with an ITA value; a group passes if
|group MAE − population MAE| ≤ τ = 2 bpm (fixed in `config/c2_config.json` before any result).
MAE (deviation from population):

| ITA group | n | `pos` | `pos_kf` | `pos_snr` | **`pos_kf_snr`** (95 % CI) |
|---|---|---|---|---|---|
| I–II (ITA > 41°) | 15 | 17.71 (+1.65) | 10.80 (+0.95) | 9.33 (+1.21) | **5.37 (−0.29)** (3.3–7.5) |
| III–IV (10–41°) | 14 | 21.72 (**+5.66 fail**) | 12.15 (**+2.30 fail**) | 9.09 (+0.97) | **4.66 (−1.00)** (2.8–6.8) |
| V–VI (≤ 10°) | 56 | 14.20 (−1.86) | 9.02 (−0.83) | 7.56 (−0.57) | **5.98 (+0.33)** (4.8–7.2) |
| Population | 85 | 16.06 | 9.85 | 8.12 | **5.66** |
| All groups pass? | | no | no | yes | **yes** |

- `pos_kf_snr`: every group within ±1.0 bpm of the population MAE, overlapping 95 % CIs, and no
  trend with skin tone (Spearman ITA vs MAE ρ = −0.03, p = 0.82; baseline ρ = +0.32, p = 0.003).
- Per country, `pos` → `pos_kf_snr`: India I–II (n = 2) 8.12 → 3.53, III–IV (8) 23.95 → 4.78,
  V–VI (32) 14.29 → 6.25; Sierra Leone I–II (13) 19.19 → 5.65, III–IV (6) 18.75 → 4.50,
  V–VI (24) 14.09 → 5.62.
- Only recordings with consistent ITA and no white-balance warning (27): V–VI (n = 21) `pos` 10.07,
  `pos_snr` 5.59, `pos_kf_snr` 6.16 MAE (I–II and III–IV have 3 each). On this subset the
  combination is slightly behind SNR fusion alone.

### Conclusion

**Kalman denoising → per-region POS → SNR fusion** lowers AiPEX MAE from 16.0 to 5.8 bpm (53 %
within 5 bpm, bias +0.9), in both countries, and every ITA group is within ±1 bpm of the
population MAE. **No evidence of a performance loss for darker skin.** Caveats:
- With the combination the darkest group is no longer the lowest-error group (+0.3 bpm); the
  difference is well inside τ and the confidence intervals overlap.
- ITA from raw video depends on exposure and white balance. Several Sierra Leone recordings read
  "Very light" because of colour casts (b\* ≈ 1–2), so the light groups are partly mislabelled
  lower-quality recordings; the dark group (56) is the reliable part.
- AiPEX crops have no pixel counts and no background: the Kalman filter is unweighted there (a fixed
  low-pass) and the background reference is not tested.
- One mean HR per recording: only the recording-level estimate is validated, not tracking.

## Results: other datasets (mean subject MAE, bpm; `--face-module simple`, to be rerun with the tracker)

| Dataset | n | `pos` | `pos_bg` | `pos_eq` | `pos_snr` | `pos_snr_bg` |
|---|---|---|---|---|---|---|
| UBFC | 12 | 2.20 | 1.62 | 1.94 | 2.10 | 1.76 |
| CLBP | 5 | 10.30 | 1.86 | 10.30 | 10.34 | 2.43 |

- Python port vs MATLAB (`pos`, UBFC subjects 1/3/4): 1.53 / 3.23 / 2.61 vs 1.47 / 2.84 / 2.38.
- UBFC subject48: 12 windows locked at ~50 bpm (lighting); 8.64 → 1.39 with `pos_bg`.
- CLBP Subject068: a constant 2.0 Hz (120 bpm) lighting rhythm in face, wall and hijab; 42.97 →
  1.00 with `pos_bg`. SNR weighting alone cannot fix it (all regions share the flicker).
- MPU-rPPG (5): ~20 bpm for every method and every ITA. Subjects type/game at a desk; an RGB
  keyboard lights the face (~55–59 bpm rhythm in face, shirt and curtain); `Output.mp4` is
  1.5 Mbps. Not used for skin-tone conclusions. Reference = oximeter HR column (HR from the
  exported PPG column disagreed by 11–45 bpm on recordings 8, 9, 10).
- ITA in MPU is biased low (underexposure, warm cast: b\* ≈ 22), in UBFC high (b\* ≈ 6).

## Known limitations

- ITA depends on camera exposure and white balance. Inside one dataset it ranks subjects reliably;
  across datasets the absolute class can be off (UBFC b\* ≈ 6 pushes everyone to "Very light",
  MPU b\* ≈ 22 pushes East-Asian subjects to "Brown"). `wb_warning` flags b\* < 3.
- Few full-face subjects with dark skin. Most of the V–VI evidence comes from AiPEX.
