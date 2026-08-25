# Training

All runs fine-tune the official **SONIC release checkpoint**
(`sonic_release/last.pt`, NVIDIA GEAR-SONIC) with SONIC's `train_agent_trl.py`.

## Environment (`code/setup_zest.py`, Modal app)

- ubuntu:24.04, Python 3.11
- Isaac Sim `isaacsim[all,extscache]==5.1.0` (`--extra-index-url pypi.nvidia.com`)
- torchaudio 2.7.0 (cu128), Isaac Lab v2.3.2, `pip install -e gear_sonic/[training]`
- GPU H200 (training) / L40S (rendering); Modal volume `sonic-zest`.
- Credentials are **not committed**: create the Modal Secret
  `huggingface-secret` (key `HF_TOKEN`) and provide `GITHUB_TOKEN` as an
  environment variable for the transfer scripts


## Command

```bash
python -m modal run setup_zest.py::train_v1 \
  --num-iters 3000 --num-envs 4096 --motion zest_cartwheel_full --tag <TAG> \
  [--ckpt /mnt/sonic-data/logs_rl/<prev-run>/.../last.pt] \
  [--anchor-w 1.5] [--relax-terminations true|false]
```

## Termination configuration (the important part)

Two **separate** relaxation decisions:

- **Training** (`relax_terminations` in `train_v1`):
  - `relax_terminations=false` keeps the experiment's original adaptive/strict
    termination group (`tracking/base_adaptive_strict_ori_foot_xyz`, thresholds
    0.15–0.2 m). This is the **historical Stage 1** configuration; with it,
    episodes terminated inside the flip phase and the policy never received
    reward past t ≈ 3.5 s.
  - `relax_terminations=true` (the default of the helper) raises all four
    tracking thresholds to 100 and disables adaptation, so episodes run the
    full 353 samples. **Stage 2 and Stage 3 used this.**
- **Evaluation**: a separate set of thresholds (see `docs/EVALUATION.md`); for
  the final measurements **all four rows** (stock / 3000 / 2000 / final) use
  the same full-horizon relaxed evaluation thresholds.

## Schedule

| Stage | Content | Iters | Envs | GPU | relax_terminations |
|---|---|---|---|---|---|
| Stage 1 (3000) | full reference, from stock | 3000 | 4096 | H200 | false (original strict group) |
| Stage 2 (2000) | resume Stage 1 | 2000 | 4096 | H200 | **true** |
| Stage 3 (1500, final) | resume Stage 2 | 1500 | 4096 | H200 | true + `--anchor-w 1.5` |

Note: Stage 1 is reproducible with
`train_v1 --relax-terminations false`; the reproduction of Stage 1 requires the
environment version/checkpoint state listed in this document, so the numbers in
`docs/EVALUATION.md` are the authoritative record.

## Wall-clock / cost

- Stage 1: 3.15 h; Stage 2: 2.25 h; Stage 3: 1.7 h.
- ≈ 3.9–4.9 s per iteration at 4096 envs on H200 (~8.3 s/iter projected for
  the same motion on L40S with 2048 envs).
