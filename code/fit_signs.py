# -*- coding: utf-8 -*-
"""拟合 URDF(h5) -> MJCF(SONIC) 关节角符号约定.

原理: h5 link 位置是 URDF 模型的输出; 同一机械关节两部模型只差 ± 轴约定 (单自由度).
用大幅动作帧 (预备/翻转/落地) 的 FK 位置残差, 对 29 个关节做贪心符号搜索:
依次尝试翻转每个关节的符号, 保留残差更小的方案, 两轮收敛.
输出: data/signs.npy (29,) +1/-1, 并打印拟合后的逐 body 误差.
"""
import os

import h5py
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5 = os.path.join(os.environ['ZEST_ROOT'], 'reference_motions_g1/23_cartwheel.h5')
MJ_TO_IL = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
                     21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], np.int32)
IL_TO_MJ = np.zeros(29, np.int32)
IL_TO_MJ[MJ_TO_IL] = np.arange(29)

FPS = 50


def main():
    os.chdir(ROOT)
    with h5py.File(H5, 'r') as f:
        g = f['robot_data/trajectory_data']
        bl = [x.decode() for x in g['bodies/labels'][...]]
        t = g['time'][...].astype(np.float64)
        pos = g['bodies/pos_w'][...].astype(np.float64)
        quat = g['bodies/quat_w'][...].astype(np.float64)
        jp = g['joints/pos'][...].astype(np.float64)
    bidx = {n: i for i, n in enumerate(bl)}

    # 重采样到 50Hz (位置/关节/四元数)
    tg = np.arange(0, t[-1] + 1e-6, 1.0 / FPS)
    prs = np.zeros((len(tg), pos.shape[1], 3))
    qrs = np.zeros((len(tg), pos.shape[1], 4))
    jrs = np.zeros((len(tg), 29))
    for i, tt in enumerate(tg):
        k = np.searchsorted(t, tt, side='right') - 1
        k = max(0, min(len(t) - 2, k))
        u = (tt - t[k]) / max(1e-9, t[k + 1] - t[k])
        prs[i] = pos[k] * (1 - u) + pos[k + 1] * u
        jrs[i] = jp[k] * (1 - u) + jp[k + 1] * u
        dot = float(np.sum(quat[k] * quat[k + 1]))
        qb = quat[k + 1] if dot >= 0 else -quat[k + 1]
        v = quat[k] * (1 - u) + qb * u
        qrs[i] = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    # 采样帧: 站立(小角度) + 预备 + 翻转段(大角度, 判别力强)
    frame_ts = [0.3, 1.0, 2.0, 2.5, 2.9, 3.2, 3.5, 4.0, 5.0]
    frames = [int(np.argmin(np.abs(tg - x))) for x in frame_ts]

    m = mujoco.MjModel.from_xml_path('data/g1_kin.xml')
    d = mujoco.MjData(m)
    bids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
    targets = [(nm, 0.3 if 'wrist' in nm else (0.6 if ('torso' in nm or 'waist' in nm) else 1.0))
               for nm in bids if nm in bidx and nm not in
               ('base', 'pelvis', 'left_wrist_roll_link', 'right_wrist_roll_link',
                'left_wrist_pitch_link', 'right_wrist_pitch_link')]
    tgt_ids = np.array([bids[nm] for nm, _ in targets], np.int64)
    tgt_w = np.array([w for _, w in targets])
    tgt_names = [nm for nm, _ in targets]

    def residual(signs):
        tot = 0.0
        for fi in frames:
            d.qpos[:] = 0
            d.qpos[:3] = prs[fi, bidx['pelvis']]
            d.qpos[3:7] = qrs[fi, bidx['pelvis']]
            d.qpos[7:] = signs * jrs[fi][IL_TO_MJ]
            mujoco.mj_forward(m, d)
            e = d.xpos[tgt_ids] - prs[fi][[bidx[nm] for nm in tgt_names]]
            tot += float(np.sum((tgt_w[:, None] * e) ** 2))
        return tot

    signs = np.ones(29, np.float64)
    best = residual(signs)
    print(f'initial all-+ residual {best:.1f}')
    for rnd in range(2):
        improved = False
        for j in range(29):
            signs[j] *= -1
            cur = residual(signs)
            if cur < best:
                best = cur
                improved = True
                print(f'  round{rnd} flip joint {j:2d} -> residual {cur:.1f}')
            else:
                signs[j] *= -1
        if not improved:
            break
    print(f'\nfinal residual {best:.1f}')
    print('signs:', ''.join('+' if s > 0 else '-' for s in signs))

    # 拟合后逐 body 最大误差 @ 采样帧
    errs = np.zeros((len(frames), len(targets)))
    for i, fi in enumerate(frames):
        d.qpos[:] = 0
        d.qpos[:3] = prs[fi, bidx['pelvis']]
        d.qpos[3:7] = qrs[fi, bidx['pelvis']]
        d.qpos[7:] = signs * jrs[fi][IL_TO_MJ]
        mujoco.mj_forward(m, d)
        for k, nm in enumerate(targets):
            errs[i, k] = np.linalg.norm(d.xpos[tgt_ids[k]] - prs[fi, bidx[nm[0]]])
    worst = errs.max(axis=0)
    print('\nper-body max err @ sampled frames (cm):')
    for k in np.argsort(-worst)[:8]:
        print(f'  {targets[k][0]:28s} {worst[k]*100:6.1f}')

    np.save(os.path.join(ROOT, 'data/signs.npy'), signs)


if __name__ == '__main__':
    main()
