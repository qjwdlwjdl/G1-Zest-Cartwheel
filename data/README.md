# Data — derived motion libraries

These files are **derived** from the ZEST dataset
(Zenodo 10.5281/zenodo.21135719, CC-BY-4.0) and are distributed under
**CC-BY-4.0** as well (see `DATA_LICENSE.md`).

| File | Description |
|---|---|
| `zest_motions.pkl` | SONIC `motion_lib`: `zest_cartwheel_full` (353 samples @ 50 Hz, ~7.1 s) + `zest_cartwheel_core` (140 samples, 2.8 s) |
| `motions_zest_cartwheel_full.pkl` | Full cartwheel motion library entry |
| `motions_zest_cartwheel_core.pkl` | Core flip entry |
| `zest_cartwheel_full.csv` / `zest_cartwheel_core.csv` | Bones-SEED flat CSV intermediate (root cm/deg + 29 DOF deg) |

`motion_lib` keys: `root_trans_offset`, `pose_aa`, `dof`, `root_rot`,
`smpl_joints`, `fps`.

- Source reference: `23_cartwheel.h5` (G1, retargeted from human MoCap),
  associated with ZEST's hardware-evaluated cartwheel experiments.
- Conversion and QC: `docs/CONVERSION.md` + `code/convert_zest.py`.
- Provenance statement: see `DATA_LICENSE.md`.

Attribution required (ZEST, DOI 10.5281/zenodo.21135719, Science Robotics
manuscript aec7695); also credit NVIDIA GEAR-SONIC/SONIC for the motion
library format and **Motion Data by Bones Studio**.
