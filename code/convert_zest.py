# -*- coding: utf-8 -*-
"""ZEST 23_cartwheel.h5 -> G1 SONIC motion_lib 转换器。

数据: ZEST (Science Robotics aec7695) G1 cartwheel 参考轨迹, 120Hz HDF5,
      URDF link 帧世界系位姿 ([w,x,y,z]), 关节序为 IsaacLab 序 (il)。
约定差异: ZEST 为 URDF 约定, SONIC 用 MJCF (g1_29dof_rev_1_0.xml) 约定,
      部分关节 (膝/踝/肩/肘/腕) 轴/符号不同。
方法: 以 h5 的 link 位置为 FK 目标, 在 g1_kin.xml (SONIC 同拓扑) 上做逐帧 29-DOF 位置 IK,
      IK 结果即 SONIC 的 MJCF 约定关节角, 自动吸收全部轴/符号差异。
根: h5 pelvis 与 SONIC 自由关节 base 共点 (mjcf pos=0 0 0) -> 根位姿直接用;
      yaw 归一化使运动起始朝向接近单位 (与 DM 动作族一致), 起始 xy 归零。
外部依赖 (环境变量, 均必填):
      SONIC_ROOT    GR00T-WholeBodyControl 检出目录 (提供 gear_sonic/data_process/
                    convert_soma_csv_to_motion_lib.py 及其 MJ_TO_IL 常量)
      ZEST_ROOT     ZEST_G1_Data_Zenodo 检出目录 (reference_motions_g1/23_cartwheel.h5)
      G1_KIN_XML    g1_kin.xml 绝对路径 (SONIC 同拓扑 kinematics-only 模型)
输出 (与仓库 data/ 布局一致, 无需再整理):
      data/zest_cartwheel_{full,core}.csv  (Bones-SEED 平铺: cm/deg)
      data/qc_target_zest_cartwheel_*.npz  (IK 目标/根轨迹, QC 用)
      gear_sonic convert_soma_csv_to_motion_lib.py (--fps 50)
      -> data/zest_motions.pkl (两条目) + data/motions_zest_cartwheel_{full,core}.pkl
变体: zest_cartwheel_full (0-7.1s), zest_cartwheel_core (1.9-4.7s 核心翻转段)
"""
import os
import subprocess
import sys

import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as SROT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONIC_ROOT = os.environ.get('SONIC_ROOT', '')
ZEST_ROOT = os.environ.get('ZEST_ROOT', '')
KIN_XML = os.environ.get('G1_KIN_XML', '')
if not (SONIC_ROOT and ZEST_ROOT and KIN_XML):
    raise RuntimeError(
        '需要环境变量: SONIC_ROOT (GR00T-WholeBodyControl 检出目录), '
        'ZEST_ROOT (ZEST_G1_Data_Zenodo 检出目录), G1_KIN_XML (g1_kin.xml 绝对路径)')

sys.path.insert(0, os.path.join(SONIC_ROOT, 'gear_sonic', 'data_process'))
from convert_soma_csv_to_motion_lib import MJ_TO_IL, BONES_CSV_JOINT_NAMES  # noqa: E402
CONVERTER = os.path.join(SONIC_ROOT, 'gear_sonic', 'data_process',
                         'convert_soma_csv_to_motion_lib.py')

H5 = os.path.join(ZEST_ROOT, 'reference_motions_g1/23_cartwheel.h5')
CSV_DIR = os.path.join(ROOT, 'data')            # 输出 = 仓库已发布的 data/ 布局
OUT_PKL = os.path.join(ROOT, 'data', 'zest_motions.pkl')
FPS = 50

IL_TO_MJ = np.zeros(29, np.int32)
IL_TO_MJ[MJ_TO_IL] = np.arange(29)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def qz_quat(theta):
    return np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])


def quat_mul(a, b):  # wxyz
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def nlerp_quats(qa, qb, u):
    dot = float(np.dot(qa, qb))
    qb = qb if dot >= 0 else -qb
    v = qa * (1 - u) + qb * u
    return v / (np.linalg.norm(v) + 1e-12)


def resample(t, vals, tg, quat_cols):
    """按真实时间戳重采样; quat_cols 为四元数列起始索引(4 列一组)."""
    out = np.zeros((len(tg), vals.shape[1]), dtype=np.float64)
    for i, tt in enumerate(tg):
        idx = np.searchsorted(t, tt, side='right') - 1
        idx = max(0, min(len(t) - 2, idx))
        a, b = idx, idx + 1
        u = (tt - t[a]) / max(1e-9, t[b] - t[a]) if t[b] > t[a] else 0.0
        out[i] = vals[a] * (1 - u) + vals[b] * u
        for s in quat_cols:
            out[i, s:s + 4] = nlerp_quats(vals[a, s:s + 4], vals[b, s:s + 4], u)
    return out


def yaw_of(q):  # q = [w,x,y,z]
    return np.arctan2(2 * (q[3] * q[0] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def norm_yaw(seg_pos, seg_quat, ref_q):
    """整体绕世界 z 预旋转, 使起始朝向接近单位 (yaw), z 高度不变; 起始 xy 归零."""
    qn = qz_quat(-yaw_of(ref_q))
    Rn = quat_to_mat(qn)
    p2 = np.einsum('ij,tnj->tni', Rn, seg_pos)
    q2 = np.empty_like(seg_quat)
    w1, x1, y1, z1 = qn
    w2, x2, y2, z2 = seg_quat[..., 0], seg_quat[..., 1], seg_quat[..., 2], seg_quat[..., 3]
    q2[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q2[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q2[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q2[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    p2 -= p2[0:1] * np.array([1.0, 1.0, 0.0])
    return p2, q2


def main():
    os.chdir(ROOT)  # mujoco 相对路径加载
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)

    with h5py.File(H5, 'r') as f:
        g = f['robot_data/trajectory_data']
        bl = [x.decode() for x in g['bodies/labels'][...]]
        t = g['time'][...]
        pos = g['bodies/pos_w'][...].astype(np.float64)    # (T,40,3)
        quat = g['bodies/quat_w'][...].astype(np.float64)  # (T,40,4) wxyz
        jp = g['joints/pos'][...].astype(np.float64)       # (T,29) il 序 rad
    bidx = {n: i for i, n in enumerate(bl)}

    # 重采样到 50Hz (四元数列: 每 body 4 列, 共 40 组)
    nq_cols = quat.shape[1]  # 40 bodies
    tg = np.arange(0, t[-1] + 1e-6, 1.0 / FPS)
    qrs = resample(t, quat.reshape(len(t), -1), tg, range(0, nq_cols * 4, 4)).reshape(len(tg), nq_cols, 4)
    prs = resample(t, pos.reshape(len(t), -1), tg, []).reshape(len(tg), 40, 3)
    jrs = resample(t, jp, tg, [])
    print(f'resampled: {len(tg)} frames @{FPS}Hz ({len(tg) / FPS:.2f}s)')

    # ---- IK (g1_kin, SONIC 拓扑) ----
    m = mujoco.MjModel.from_xml_path(KIN_XML)
    d = mujoco.MjData(m)
    if m.nu != 29:
        raise RuntimeError(f'unexpected nu {m.nu}')
    bids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
    jnt_range = m.jnt_range
    lb = np.array([jnt_range[m.actuator_trnid[a][0]][0] for a in range(m.nu)])
    ub = np.array([jnt_range[m.actuator_trnid[a][0]][1] for a in range(m.nu)])
    lb = np.where(np.isfinite(lb), lb, -np.pi)
    ub = np.where(np.isfinite(ub), ub, np.pi)

    # 目标 body: kin 模型与 h5 共有; 腕 roll/pitch link 帧 URDF/MJCF 不一致, 仅盯 wrist_yaw link;
    # 腕部目标降权 (腕自由度固定, 位置由固定角决定)
    targets = [(nm, 0.3 if 'wrist' in nm else (0.6 if ('torso' in nm or 'waist' in nm) else 1.0))
               for nm in bids if nm in bidx and nm not in
               ('base', 'pelvis', 'left_wrist_roll_link', 'right_wrist_roll_link',
                'left_wrist_pitch_link', 'right_wrist_pitch_link')]
    tgt_ids = np.array([bids[nm] for nm, _ in targets])
    tgt_w = np.array([w for _, w in targets])
    # 腕 3+3 DOF (MJ idx 19-21/26-28) 固定为 h5 值 (URDF/MJCF 手端 link 帧不同, 无法经 IK 对齐);
    # 符号来自 64 组合扫描最优: 仅右腕 yaw 翻转
    WR_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
    FREE_IDX = np.array([i for i in range(29) if i not in (19, 20, 21, 26, 27, 28)])
    FIXED_IDX = np.array([i for i in range(29) if i not in FREE_IDX])
    print(f'IK targets: {len(targets)} bodies')

    def solve_seg(i0, i1):
        """逐帧 IK (23 自由 + 腕 6 固定); 根取 h5 pelvis (与自由关节共点).

        每帧以 h5 原始角为热启动 (URDF/MJCF 只差微小约定, 起点天然在正确分支;
        腕部固定不参与优化, 不存在 2π 累积)。解仅钳制回界, 不做跨帧分支改写。
        """
        qpos_out = np.zeros((i1 - i0, 36))
        costs = np.zeros(i1 - i0)
        for ti, i in enumerate(range(i0, i1)):
            tgts = np.array([prs[i][bidx[nm]] for nm, _ in targets])
            qfix = np.clip(jrs[i][IL_TO_MJ][FIXED_IDX] * WR_SIGNS, lb[FIXED_IDX], ub[FIXED_IDX])
            d.qpos[:] = 0
            d.qpos[:3] = prs[i][bidx['pelvis']]
            d.qpos[3:7] = qrs[i][bidx['pelvis']]
            x = np.clip(jrs[i][IL_TO_MJ][FREE_IDX], lb[FREE_IDX], ub[FREE_IDX])

            def resid(xq):
                qfull = np.zeros(29)
                qfull[FREE_IDX] = xq
                qfull[FIXED_IDX] = qfix
                d.qpos[7:] = qfull
                mujoco.mj_forward(m, d)
                return (tgt_w[:, None] * (d.xpos[tgt_ids] - tgts)).ravel()

            sol = least_squares(resid, x, bounds=(lb[FREE_IDX], ub[FREE_IDX]),
                                max_nfev=150, ftol=1e-6, xtol=1e-6, verbose=0)
            q = np.zeros(29)
            q[FIXED_IDX] = qfix
            q[FREE_IDX] = np.clip(sol.x, lb[FREE_IDX], ub[FREE_IDX])
            qpos_out[ti, :3] = d.qpos[:3]
            qpos_out[ti, 3:7] = qrs[i][bidx['pelvis']]
            qpos_out[ti, 7:] = q
            costs[ti] = sol.cost
            if ti % 40 == 0:
                print(f'  frame {ti}/{i1 - i0}  cost {sol.cost:.4e}', flush=True)
        return qpos_out, costs

    def postproc(qpos_out):
        qd = qpos_out[:, 7:].copy()
        for c in range(29):
            qd[:, c] = np.unwrap(qd[:, c], period=2 * np.pi)
            if len(qd) > 9:
                qd[:, c] = savgol_filter(qd[:, c], window_length=9, polyorder=2)
            # 折叠回界限中心区间 (G1 限位跨角均 < 2π, 姿势等价且数值入限)
            mids = 0.5 * (lb[c] + ub[c])
            qd[:, c] = mids + np.arctan2(np.sin(qd[:, c] - mids), np.cos(qd[:, c] - mids))
            qd[:, c] = np.clip(qd[:, c], lb[c], ub[c])
            # 连续性修复: 折叠产生的限位边界大跳变限幅 (保持姿势等价, 消除速度尖峰)
            for _ in range(8):
                dv = np.diff(qd[:, c])
                if np.max(np.abs(dv)) <= 1.5:
                    break
                for i in np.where(np.abs(dv) > 1.5)[0]:
                    qd[i + 1, c] = qd[i, c] + np.clip(dv[i], -1.5, 1.5)
            qd[:, c] = np.clip(qd[:, c], lb[c], ub[c])
        qpos_out[:, 7:] = qd
        return qpos_out

    def build_variant(name, i0, i1):
        qpos_out, costs = solve_seg(i0, i1)
        qpos_out = postproc(qpos_out)
        p_n, q_n = norm_yaw(prs[i0:i1], qrs[i0:i1], qrs[i0, bidx['pelvis']])
        root_p = p_n[:, bidx['pelvis']]
        qc = q_n[:, bidx['pelvis']]
        eul = SROT.from_quat(qc[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)

        cols = ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
                'root_rotateX', 'root_rotateY', 'root_rotateZ'] + BONES_CSV_JOINT_NAMES
        lines = [','.join(cols)]
        dof_deg = np.degrees(qpos_out[:, 7:])
        for ti in range(len(root_p)):
            row = [ti] + list(np.round(root_p[ti] * 100, 4)) + list(np.round(eul[ti], 3)) + \
                  list(np.round(dof_deg[ti], 3))
            lines.append(','.join(str(x) for x in row))
        csv_p = os.path.join(CSV_DIR, f'{name}.csv')
        with open(csv_p, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))

        np.savez(os.path.join(ROOT, 'data', f'qc_target_zest_cartwheel_{name}.npz'),
                 t=tg[i0:i1] - tg[i0], root_p=root_p,
                 link_p=p_n[:, [bidx[nm] for nm, _ in targets]],
                 link_names=np.array([nm for nm, _ in targets]))
        print(f'[{name}] {len(root_p)} frames | mean cost {costs.mean():.4e} | CSV -> {csv_p}')
        return root_p

    build_variant('zest_cartwheel_full', 0, len(tg))
    build_variant('zest_cartwheel_core', np.searchsorted(tg, 1.9), np.searchsorted(tg, 4.7))

    # ---- 调用 gear_sonic 转换器 -> motion_lib pkl ----
    print('\n>>> gear_sonic convert_soma_csv_to_motion_lib.py')
    r = subprocess.run([sys.executable, CONVERTER,
                        '--input', CSV_DIR, '--output', OUT_PKL, '--fps', str(FPS)],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        raise RuntimeError('converter failed')

    import joblib
    pkl = joblib.load(OUT_PKL)
    for name in ('zest_cartwheel_full', 'zest_cartwheel_core'):
        e = pkl[name]
        print(f'{name}: {e["dof"].shape} frames, fps={e["fps"]}, '
              f'root z[{e["root_trans_offset"][:, 2].min():.3f},{e["root_trans_offset"][:, 2].max():.3f}], '
              f'max|dof| {np.abs(e["dof"]).max():.2f} rad')
        joblib.dump({name: e}, os.path.join(ROOT, 'data', f'motions_zest_cartwheel_{name}.pkl'),
                    compress=True)


if __name__ == '__main__':
    main()
