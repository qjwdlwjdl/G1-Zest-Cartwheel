#!/bin/bash
# Generic SONIC eval-render script (run on a GPU machine with the SONIC env installed).
#
# Usage:
#   source ~/miniconda3/bin/activate env311          # activate the environment
#   MOTION=~/motions/cartwheel.pkl \
#   CKPT=~/ckpts/final/last.pt \
#   CONFIG=~/ckpts/final/config.yaml \
#   OUTDIR=~/renders TAG=cartwheel_after \
#   bash scripts/render_generic.sh
#
# Motion pkl must contain a single motion key with "root_rot" (see data/).
# Camera: follows the robot (fix_camera_after_first_frame=false), distance DIST (default 7.0).

set -e
MOTION=${MOTION:?MOTION pkl required}
CKPT=${CKPT:?CKPT required}
CONFIG=${CONFIG:?CONFIG required}
TAG=${TAG:?TAG required}
OUTDIR=${OUTDIR:?OUTDIR required}
DIST=${DIST:-7.0}
MAX_STEPS=${MAX_STEPS:-353}
SKIP=${SKIP:-2}

mkdir -p "$OUTDIR" /tmp/render_generic_$TAG

cat > /tmp/calc_offset.py << 'PY'
import sys, joblib, numpy as np, torch
data = joblib.load(sys.argv[1])
key = list(data.keys())[0]
root_rot = data[key]["root_rot"]
q = torch.tensor(root_rot[0], dtype=torch.float64)
qq = torch.tensor([q[3], q[0], q[1], q[2]], dtype=torch.float64)
def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
v = torch.tensor([0.0,1.0,0.0,0.0])
qq_inv = torch.tensor([qq[0], -qq[1], -qq[2], -qq[3]])
fwd = qmul(qmul(qq, v), qq_inv)
f = fwd[1:].numpy()
dist = float(sys.argv[2])
off = -f * dist + np.array([0, 0, 0.55])
print("[%.3f, %.3f, %.3f]" % tuple(off))
PY

OFFSET=$(python /tmp/calc_offset.py "$MOTION" "$DIST")
echo "offset=$OFFSET"

CK=ck_$TAG
mkdir -p "$CK"
cp "$CKPT" "$CK/last.pt"
cp "$CONFIG" "$CK/config.yaml"

printf 'Yes\n' | HYDRA_FULL_ERROR=1 python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CK/last.pt" +headless=True ++num_envs=1 \
    manager_env/recorders=render ++eval_callbacks=[] \
    ++manager_env.config.render_results=true \
    ++manager_env.config.save_rendering_dir=/tmp/render_generic_$TAG \
    ++manager_env.config.fix_camera_after_first_frame=false \
    ++manager_env.config.eval_camera_offset="$OFFSET" \
    ++manager_env.config.max_render_envs=1 \
    ++manager_env.config.render_frame_skip=$SKIP \
    ++max_render_steps=$MAX_STEPS ++run_once=true \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION" \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros \
    ++use_wandb=false

for f in /tmp/render_generic_$TAG/*.mp4; do
    [ -f "$f" ] && mv "$f" "$OUTDIR/$TAG.mp4" && echo "[OK] $OUTDIR/$TAG.mp4"
done
echo RENDER_DONE
