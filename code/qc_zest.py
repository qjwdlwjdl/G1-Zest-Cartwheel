# -*- coding: utf-8 -*-
"""ZEST 转换产物 QC: motion_lib 关节角 FK 重建 vs ZEST 原始 h5 目标 (原始帧, 无归一化).

- 逐 50Hz 帧: 取 h5 插值目标 (pelvis 位姿 + link 位置), FK 用 pkl dof + 原始根位姿
- 统计各 body 平均/最大位置误差, 关节速度峰值, NaN
- 输出骨架对比图 (红=参考, 蓝=重建)
"""
import os

import h5py
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, 'data', 'zest_motions.pkl')
H5 = os.path.join(os.environ['ZEST_ROOT'], 'reference_motions_g1/23_cartwheel.h5')

SEGMENTS = [
    ('pelvis', 'torso_link'),
    ('torso_link', 'left_shoulder_pitch_link'), ('left_shoulder_pitch_link', 'left_elbow_link'),
    ('torso_link', 'right_shoulder_pitch_link'), ('right_shoulder_pitch_link', 'right_elbow_link'),
    ('left_elbow_link', 'left_wrist_yaw_link'), ('right_elbow_link', 'right_wrist_yaw_link'),
    ('left_hip_pitch_link', 'left_knee_link'), ('right_hip_pitch_link', 'right_knee_link'),
    ('left_knee_link', 'left_ankle_pitch_link'), ('right_knee_link', 'right_ankle_pitch_link'),
    ('left_ankle_pitch_link', 'left_ankle_roll_link'), ('right_ankle_pitch_link', 'right_ankle_roll_link'),
]


def main():
    os.chdir(ROOT)
    m = mujoco.MjModel.from_xml_path('data/g1_kin.xml')
    d = mujoco.MjData(m)
    bids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}

    with h5py.File(H5, 'r') as f:
        g = f['robot_data/trajectory_data']
        bl = [x.decode() for x in g['bodies/labels'][...]]
        t = g['time'][...]
        pos = g['bodies/pos_w'][...]
        quat = g['bodies/quat_w'][...]
    bidx = {n: i for i, n in enumerate(bl)}

    bodies_cmp = [nm for nm in bids if nm in bidx]
    pkl = joblib.load(PKL)
    for name, t0 in (('zest_cartwheel_full', 0.0), ('zest_cartwheel_core', 1.9)):
        e = pkl[name]
        T = e['dof'].shape[0]
        errs = np.zeros((T, len(bodies_cmp)))
        recon = np.zeros((T, len(bodies_cmp), 3))
        tgt = np.zeros((T, len(bodies_cmp), 3))
        for i in range(T):
            tt = t0 + i / 50.0
            k = np.searchsorted(t, tt, side='right') - 1
            k = max(0, min(len(t) - 2, k))
            u = (tt - t[k]) / max(1e-9, t[k + 1] - t[k])
            p = pos[k] * (1 - u) + pos[k + 1] * u
            q4 = quat[k] * (1 - u) + quat[k + 1] * u
            q4 /= np.linalg.norm(q4, axis=-1, keepdims=True)
            d.qpos[:] = 0
            d.qpos[:3] = p[bidx['pelvis']]
            d.qpos[3:7] = q4[bidx['pelvis']]
            d.qpos[7:] = e['dof'][i]
            mujoco.mj_forward(m, d)
            for j, nm in enumerate(bodies_cmp):
                recon[i, j] = d.xpos[bids[nm]]
                tgt[i, j] = p[bidx[nm]]
                errs[i, j] = np.linalg.norm(recon[i, j] - tgt[i, j])

        print(f'== {name} == T={T} frames ({T / 50:.2f}s)')
        print(f'  overall mean {errs.mean() * 100:.1f} cm | p95 {np.percentile(errs, 95) * 100:.1f} cm '
              f'| max {errs.max() * 100:.1f} cm')
        worst = np.argsort(-errs.max(axis=0))[:5]
        for j in worst:
            print(f'  {bodies_cmp[j]:28s} max {errs[:, j].max() * 100:6.1f} cm')
        jv = np.abs(np.diff(e['dof'], axis=0)) * 50.0
        print(f'  max joint vel {jv.max():.1f} rad/s | NaN={np.isnan(e["dof"]).any()} '
              f'| max|dof|={np.abs(e["dof"]).max():.2f} rad')

        frames = np.unique(np.linspace(0, T - 1, 8).astype(int))
        cidx = {nm: j for j, nm in enumerate(bodies_cmp)}
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        for ax, fi in zip(axes.ravel(), frames):
            for (a, b) in SEGMENTS:
                if a not in cidx or b not in cidx:
                    continue
                ia, ib = cidx[a], cidx[b]
                ax.plot([tgt[fi, ia, 0], tgt[fi, ib, 0]], [tgt[fi, ia, 2], tgt[fi, ib, 2]],
                        'r-', lw=2, alpha=.8)
                ax.plot([recon[fi, ia, 0], recon[fi, ib, 0]], [recon[fi, ia, 2], recon[fi, ib, 2]],
                        'b--', lw=2, alpha=.8)
            ax.set_title(f't={fi / 50:.1f}s', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(alpha=.3)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.1, 1.8)
        fig.suptitle(f'{name}: red=ZEST ref, blue=SONIC motion_lib FK (x-z)', fontsize=12)
        out = os.path.join(ROOT, f'data/qc_{name}.png')
        fig.savefig(out, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'  figure -> {out}')


if __name__ == '__main__':
    main()
