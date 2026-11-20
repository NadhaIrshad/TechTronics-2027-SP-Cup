# TechTronics — IEEE SP Cup 2027 (FaceVitals)

2027 SP Cup, Team TechTronics, Department of Electronic and Telecommunication Engineering, University of Moratuwa, Sri Lanka.

**Challenge:** contactless estimation of Heart Rate, SpO₂ and Blood Pressure from RGB facial video (remote photoplethysmography, rPPG).
**Round 1:** Heart Rate only. Deadline **30 November 2026**.

---

## Pipeline

```
Video ─► (Constraint simulation) ─► Face detection & tracking ─► ROI + skin mask ─► RGB traces ─► rPPG signal ─► HR ─► Metrics
```

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
│   ├── io/                 # video / ground-truth loaders, data export
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

## Datasets

- **UBFC-rPPG:** https://sites.google.com/view/ybenezeth/ubfcrppg
- **AiPEX rppg_biases:** https://github.com/AiPEX-Lab/rppg_biases
- **Organizer dataset (Sri Ramachandra):** covered by a data agreement. Keep it on team machines only.

Put the datasets under `data/raw/`. **Do not commit videos or `.mat` data files.** The `data/` and `outputs/` folders are gitignored.
