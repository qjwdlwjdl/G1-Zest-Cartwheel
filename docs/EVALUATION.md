# Evaluation

## Protocol

- `eval_agent_trl.py` with `eval_callbacks=im_eval`, `num_envs=32`
  (32 rollouts per checkpoint), `run_eval_loop=False`.
- Motion: `zest_cartwheel_full.pkl` (353 samples @ 50 Hz, ~7.1 s; the recorded
  evaluation window is 352 steps).
- **Full-window fix (evaluation side)**: the default eval termination group
  (`tracking/eval`: anchor_pos 0.25 m, anchor_ori_full 1.0 rad, ee_body_pos
  0.25 m, foot_pos_xyz 0.5 m) terminated **every** episode at t ≈ 3.5 s
  (logs show `Terminated: N` at steps 176–178, 32/32 environments). All
  reported metrics below use thresholds raised to 100, so all episodes run the
  whole 352-step window to time-out — **for every row identically**.
- Metrics (defined precisely):
  - **mpjpe_g**: mean over rollout timesteps of the mean per-body Euclidean
    position error, averaged over all 32 rollouts.
  - **mpjpe_l**: same, but with the root (pelvis) removed from both tracks
    before computing distances.
  - **root drift**: **mean Euclidean root-position error**
    (`‖root_policy − root_reference‖` averaged over timesteps and rollouts).

## Ground-truth note

`im_eval` stores one array per rollout with shape `(T, 14, 3)` — 32 entries =
32 rollouts (not time-steps across environments). The skeleton visualizer must
interpret it that way.

## Results (full window, same motion, same protocol for every row)

| Policy | mpjpe_g (mm) | mpjpe_l (mm) | root drift (mm) | Outcome |
|---|---|---|---|---|
| Stock `sonic_release` | 913.7 | 294.7 | 888.3 | physically collapses at ≈2.4 s; rollout runs to full horizon |
| FT Stage 1 (3000) | 535.7 | 215.7 | 539.2 | collapses at ≈3.4 s; rollout runs to full horizon |
| FT Stage 2 (2000) | 135.4 | 39.7 | 125.6 | full window, lands |
| **FT Stage 3 (final, 1500)** | **101.5** | **41.3** | **90.5** | full window, lands |

**Best-rollout detail (final policy)**: best-rollout mean mpjpe_g = **58 mm**;
best-rollout final-frame mpjpe_g = **33 mm**; final-frame pelvis height
0.788 m (reference 0.787 m); root error by phase 31 / 48 / 54 / 18 mm.

Note the distinction: `mpjpe_g = 101.5 mm` is the average over all 32 rollouts;
`58 mm` and `33 mm` are the *best-rollout* statistics and are intentionally
labeled as such.

## Known residuals

- A minor transient ground penetration (≈ −3.8 cm) was observed in some
  rollouts (reference min z ≈ +3.5 cm).
- Conversion-time wrist link-frame difference (25–33 cm between the ZEST URDF
  and the SONIC MJCF hand frames) — a frame convention difference, not
  joint-space behavior (see `docs/CONVERSION.md`).

## Rendering

Physics renders (`manager_env/recorders=render`, `run_once=true`, follow
camera at 7 m, 25 fps / 1920×1088): see `scripts/render_generic.sh` and
`videos/zest_cw_before.mp4` / `videos/zest_cw_after_v3.mp4` /
`videos/zest_cw_before_after_captioned.mp4`.
