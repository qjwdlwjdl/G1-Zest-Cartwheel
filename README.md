# G1 Cartwheel — SONIC (GEAR-SONIC) Fine-Tune

**Project**: Teach the Unitree G1 humanoid to perform a complete **cartwheel** by
fine-tuning the NVIDIA GEAR-SONIC (SONIC) motion-imitation agent on the ZEST G1
cartwheel reference.

**Track**: Experimental

**Model policy deployed**: `model_step_001500_*` ONNX exports (see the private
Hugging Face model repository).

---

## What was taught

The policy tracks a **~7.1-second G1 cartwheel reference** (353 samples @ 50 Hz,
~2.0 m of travel) derived from ZEST's G1 cartwheel experiment:

1. standing prep
2. arm raise / entry
3. **hand-supported lateral inversion, with both legs passing overhead**
   (the wrist contact phase)
4. landing with knee absorption
5. recovery to standing

Evaluation is performed over the **full window of 352 steps / ~7.1 s** — not a
truncated prefix.

## Why it is hard

- **Hand contact**: unlike running or jumping, this move requires the wrists to
  carry the body — a contact-rich transient absent from the source (stock)
  checkpoint's training data.
- **Long horizon**: 353 samples is longer than the motions the stock checkpoint
  was released with, so the landing and recovery are easy to miss if episodes
  are cut short.
- **Root translation**: ~2.0 m of travel means root-tracking error accumulates
  over the whole sequence.
- **Evaluation dead-end**: the default evaluation termination terms (base
  height, base orientation, wrist/ankle height, foot position) terminated every
  episode at t ≈ 3.5 s, so naive before/after metrics compare only the easy
  first half. We relaxed the thresholds to run the full window and re-measured
  everything (see `docs/EVALUATION.md`).

## How it was done

1. **Reference**: ZEST `reference_motions_g1/23_cartwheel.h5` — a G1 cartwheel
   reference from ZEST, associated with ZEST's hardware-evaluated cartwheel
   experiments (Table 1 of the ZEST manuscript: MAE(q) = 0.078 rad,
   multi-contact = yes).
2. **Conversion**: the ZEST poses were converted into the SONIC MJCF convention
   using **per-frame optimization** (position IK), because the ZEST URDF and
   the SONIC MJCF use different joint conventions; the 6 wrist DOFs were handled
   separately (fixed to the reference values with a fitted sign for the right
   wrist yaw) because the two hand link frames differ. See
   `docs/CONVERSION.md` for the exact procedure, QC numbers, and the documented
   wrist link-frame residual (25–33 cm under hand support).
   - **Reproduction prerequisites**: conversion requires an external checkout of
     the ZEST dataset (`reference_motions_g1/23_cartwheel.h5`), the G1 MJCF
     assets (`data/g1_kin.xml`, `g1_29dof_rev_1_0.xml`) and a SONIC /
     IsaacLab installation. See `docs/CONVERSION.md`.
3. **Training**: `code/setup_zest.py` (Modal app; Isaac Sim 5.1.0 + Isaac Lab
   v2.3.2 + gear_sonic; fine-tunes from the SONIC `sonic_release` checkpoint).
   Staged schedule: 3000 iterations on the full reference, +2000 with relaxed
   episode terminations, +1500 with an increased root-position reward weight.
4. **Evaluation**: 32 rollouts per checkpoint over the full 352-step window,
   with the same relaxed evaluation thresholds for all rows; metrics
   mpjpe_g / mpjpe_l / root drift; physical renders on an L40S
   (`scripts/render_generic.sh`).

### Results (full ~7.1 s window, 32 rollouts, identical evaluation protocol for every row)

| Policy | mpjpe_g (mm) | mpjpe_l (mm) | root drift (mm) | Outcome |
|---|---|---|---|---|
| Stock `sonic_release` | 913.7 | 294.7 | 888.3 | physically collapses at ≈2.4 s; rollout runs to full horizon |
| FT, full ref, 3000 it (H200) | 535.7 | 215.7 | 539.2 | collapses at ≈3.4 s; rollout runs to full horizon |
| FT + 2000 it, relaxed terms | 135.4 | 39.7 | 125.6 | full window, lands |
| **Final: + 1500 it, anchor w=1.5** | **101.5** | **41.3** | **90.5** | **full window, lands** |

- mpjpe_g / mpjpe_l / root drift are **mean-over-all-rollouts** errors
  (root drift = mean Euclidean root-position error).
- Final policy, **best rollout**: best-rollout mean mpjpe_g = **58 mm** over the
  full window; best-rollout final-frame mpjpe_g = **33 mm**; final pelvis height
  0.788 m vs reference 0.787 m — the best episode performs the whole cartwheel
  and returns to standing.
- Rendered before/after: `videos/zest_cw_before.mp4` (stock — collapses),
  `videos/zest_cw_after_v3.mp4` (**final policy — complete move, landing**),
  `videos/zest_cw_before_after_captioned.mp4` (side-by-side with phase
  captions), `videos/zest_cw_before_after_side.mp4` (plain side-by-side).

## Repository layout

```
code/            conversion, QC, training app (convert_zest.py / qc_zest.py /
                 fit_signs.py / setup_zest.py)
scripts/         generic render script for recorded evaluations
data/            derived motion libraries (SONIC motion_lib pkl + Bones-SEED CSV)
                 — see data/README.md + DATA_LICENSE.md (CC-BY-4.0 attribution)
docs/            CONVERSION / TRAINING / EVALUATION notes + QC figures
videos/          before/after renders
```

## Reproduction notes

- Conversion: `python code/convert_zest.py` (requires the external prerequisites
  listed in `docs/CONVERSION.md`; the repo does **not** bundle the ZEST HDF5 or
  the G1 MJCF files).

## Credits / attribution

- **ZEST dataset** (reference motions): "ZEST: Zero-shot Embodied Skill Transfer
  for Athletic Robot Control" (Science Robotics, manuscript aec7695), Zenodo
  record **10.5281/zenodo.21135719**, **CC-BY-4.0**. The derived motion
  libraries in `data/` carry their own attribution/license notes
  (`data/DATA_LICENSE.md`).
- **SONIC / GEAR-SONIC**: NVIDIA, https://github.com/NVlabs/GR00T-WholeBodyControl
  (base checkpoint `nvidia/GEAR-SONIC`).
- **Isaac Lab / Isaac Sim**: NVIDIA, https://github.com/isaac-sim/IsaacLab
  (v2.3.2, Isaac Sim 5.1.0).
- **Unitree G1** robot model (`g1_29dof_rev_1_0.xml`).
- **Motion Data by Bones Studio**.
- Compute: NVIDIA H200 / L40S GPUs via Modal and Nebius AI Studio.
