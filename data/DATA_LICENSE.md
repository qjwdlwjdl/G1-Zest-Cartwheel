# DATA LICENSE — derived motion libraries

**License: CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/)

The `.pkl` / `.csv` files in this directory are **derived works** of:

> **ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control**
> (Science Robotics, manuscript aec7695)
> Dataset record: https://zenodo.org/records/21135719 · DOI: 10.5281/zenodo.21135719
> License: CC-BY-4.0

Derivation performed in this project (see `docs/CONVERSION.md`): resampling of
the ZEST G1 cartwheel reference (120 Hz → 50 Hz), conversion of the ZEST URDF
convention into the SONIC MJCF convention via per-frame position IK (wrist DOFs
handled separately, fitted sign), smoothing/continuity post-processing, and
re-formatting into the SONIC `motion_lib` pickle schema. The joint angles
themselves come from the ZEST reference; no new motion capture was performed.

When using these files, you must at minimum refer to the ZEST dataset as above
and keep the CC-BY-4.0 license; also credit **Motion Data by Bones Studio**.

Note: this license applies to the **data** in this directory. The code in the
repository is governed separately (see each code file / repository licensing
statement; the SONIC/GEAR-SONIC, Isaac Lab and Unitree assets retain their own
licenses).
