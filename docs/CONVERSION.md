# Conversion: ZEST HDF5 → SONIC motion library

## External prerequisites (not bundled in this repo)

The ZEST HDF5 dataset, the G1 MJCF assets and the SONIC checkout are
**external**; point the converter at them with environment variables:

| Variable | Path (example) |
|---|---|
| `SONIC_ROOT` | `/path/to/GR00T-WholeBodyControl` — provides `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` (and its `MJ_TO_IL` / `BONES_CSV_JOINT_NAMES` constants, which the converter imports) |
| `ZEST_ROOT` | `/path/to/ZEST_G1_Data_Zenodo` — provides `reference_motions_g1/23_cartwheel.h5` |
| `G1_KIN_XML` | `/path/to/g1_kin.xml` — SONIC-topology kinematics-only MJCF used for the IK |
| (tooling) | a SONIC / IsaacLab Python environment with mujoco, scipy, joblib |

The repo does **not** bundle the ZEST HDF5, `g1_kin.xml` or
`g1_29dof_rev_1_0.xml` (G1 MJCF from Sony/Unitree assets; see SONIC).

## What the converter does

1. **Resample** the 120 Hz HDF5 joint/root data to 50 Hz (quaternion nlerp on
   `quat_w`, per-native time base).
2. **Root pose**: taken directly from the HDF5 pelvis (`bodies/pelvis`) world
   position/orientation (the SONIC MJCF coincides the free joint base with the
   pelvis), yaw-normalized to the first frame and XY-zeroed at the start.
3. **Joint angles**: *not* a 1:1 mapping. The ZEST URDF and the SONIC MJCF use
   different joint/convention frames, so the 23 non-wrist DOFs are solved
   **per frame with position IK** (least-squares on 14 body targets, warm-started
   from the resampled reference). The **6 wrist DOFs** are fixed to the
   resampled reference values — with a fitted sign on the right wrist yaw —
   because the two hand link frames differ geometrically (25–33 cm under hand
   support) and including them in the IK causes 2π branch artifacts.
4. **Post-processing**: angle unwrap, outlier repair, Savitzky-Golay smoothing,
   fold to joint limits, continuity clamp.
5. **Output**: Bones-SEED flat CSVs **directly into `data/`** — the same
   layout that is committed in this repository — then the official
   `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` (50 Hz) is
   invoked to produce the SONIC motion_lib pkl files.

## Outputs (identical to the layout committed in `data/`)

```
data/zest_cartwheel_full.csv          # Bones-SEED flat CSV (cm/deg), full ~7.1 s
data/zest_cartwheel_core.csv          # core inversion segment (1.9–4.7 s)
data/zest_motions.pkl                 # SONIC motion_lib, both entries
data/motions_zest_cartwheel_full.pkl  # per-entry pickles
data/motions_zest_cartwheel_core.pkl
data/qc_target_zest_cartwheel_*.npz   # IK targets / root trajectory (QC)
```

No post-conversion copy/rename is needed — rerunning the converter
regenerates the committed data exactly.

## QC

FK reconstruction of the converted motion vs. the raw resampled reference:

| Entry | mean | p95 | max (wrist links only) | max\|dof\| |
|---|---|---|---|---|
| full | 1.0 cm | 5.1 cm | 32.7 cm | 3.09 rad |
| core | 1.4 cm | 8.8 cm | 32.7 cm | 3.09 rad |

**Known residual**: the wrist link-frame difference above is a URDF-vs-MJCF
frame convention difference; it affects wrist link geometry (not joint-space
behavior) and is documented here for transparency.

## Commands

```bash
export SONIC_ROOT=/path/to/GR00T-WholeBodyControl
export ZEST_ROOT=/path/to/ZEST_G1_Data_Zenodo
export G1_KIN_XML=/path/to/g1_kin.xml
python code/convert_zest.py        # regenerates data/zest_cartwheel_*.csv + *.pkl
python code/qc_zest.py             # FK-reconstruction QC figures
```

Approximate runtime: a few minutes on a laptop CPU for the 353-frame motion.
