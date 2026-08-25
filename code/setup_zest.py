"""SONIC on Modal — 环境镜像定义 + 初始化函数。

复刻自上次 Nebius L40S 上验证过的 setup_cloud.sh：
  ubuntu24.04 + py3.11 / libGLU + libnvidia-gl-580（Vulkan ICD，Modal 宿主驱动同为 580.x）
  isaacsim[all,extscache]==5.1.0 / torchaudio 2.7.0 cu128
  IsaacLab v2.3.2 + GR00T-WholeBodyControl(gear_sonic)
省略：conda（镜像自带 py3.11）、SMPL 30GB 数据（训练用 zeros，不需要）。
"""
import os
import shutil
import subprocess

import modal

SONIC = "/opt/sonic"
ISAACLAB = "/opt/IsaacLab"
VOL = "/mnt/sonic-data"

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .env({
        "PIP_NO_CACHE_DIR": "1",
        "HF_HOME": f"{VOL}/hf-cache",  # HF 下载缓存进卷，跨运行复用
        "PIP_CONSTRAINT": "/opt/pip-constraints.txt",
    })
    # setuptools>=82 移除 pkg_resources，会弄坏 SMPLSim/smplx 等老 setup.py（含隔离构建环境）
    .run_commands("printf 'setuptools<81\\n' > /opt/pip-constraints.txt")
    .apt_install(
        "git", "git-lfs", "wget", "curl", "cmake", "build-essential",
        "libglu1-mesa",            # 上次关键修复：缺它 omni.kit.usd.mdl 崩溃
        "libgl1", "libegl1",       # libGL.so.1 / libEGL —— Nebius 自带，最小镜像需补
        "vulkan-tools", "util-linux", "procps",
    )
    # 注：libnvidia-gl-580-server(580.173) 与 Modal 宿主驱动(580.95)不匹配且会搞坏 apt，
    # 不装；Vulkan 渲染阶段用匹配版本单独处理
    # Isaac Sim 5.1.0（最大的一层，~20GB 下载）
    .run_commands(
        "pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com",
    )
    .run_commands(
        "pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.2 --depth 1 {ISAACLAB}",
        f"git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git {SONIC}",
        f"cd {SONIC} && git lfs install --force && git lfs pull",
    )
    .run_commands(
        f"cd {ISAACLAB} && printf 'y\\n' | script -qec './isaaclab.sh --install' /dev/null",
        # isaaclab.sh 漏装核心包；其 setup.py 依赖 pkg_resources（新版 setuptools 已移除）
        'pip install "setuptools<81" wheel',
        f"pip install -e {ISAACLAB}/source/isaaclab --no-build-isolation",
    )
    .run_commands(
        f"cd {SONIC} && pip install -e 'gear_sonic/[training]'",
        "pip install open3d vector_quantize_pytorch",
    )
    # Isaac Sim 运行时需要的 X11/glib 库（Nebius 镜像自带；放最后一层保留上游缓存）
    .run_commands(
        "apt-get update -qq && apt-get install -y --no-install-recommends"
        " libxt6 libglib2.0-0t64 libsm6 libice6 libxkbcommon0 libxrender1"
    )
)

app = modal.App("sonic-zest-env", image=image)

volume = modal.Volume.from_name("sonic-zest")
# 凭证：不落库。先在 Modal 控制台创建 Secret("huggingface-secret")，内容 HF_TOKEN=<token>
hf_secret = modal.Secret.from_name("huggingface-secret")


def restore_checkpoint() -> None:
    """卷 → 镜像层：把 SONIC checkpoint 放回训练代码期望的位置（幂等）。"""
    src = f"{VOL}/checkpoints/sonic_release"
    dst = f"{SONIC}/sonic_release"
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(f"[restore] {src} -> {dst}")
    elif os.path.exists(dst):
        print("[restore] checkpoint 已在位")
    else:
        print("[restore] 卷中没有 checkpoint（先跑 setup_models）")


@app.function(volumes={VOL: volume}, timeout=3600)
def setup_models():
    """下载 SONIC 训练 checkpoint（跳过 30GB SMPL）+ 样例数据，存入卷。"""
    ck = f"{VOL}/checkpoints"
    if os.path.exists(f"{ck}/sonic_release/last.pt"):
        print("[setup_models] 卷里已有 checkpoint，跳过下载")
        return
    os.makedirs(ck, exist_ok=True)
    os.chdir(SONIC)
    subprocess.run(
        "python download_from_hf.py --training --no-smpl", shell=True, check=True
    )
    subprocess.run("python download_from_hf.py --sample", shell=True, check=True)
    for d in ("sonic_release", "sample_data"):
        if os.path.isdir(f"{SONIC}/{d}"):
            shutil.copytree(f"{SONIC}/{d}", f"{ck}/{d}")
            print(f"[setup_models] 已存入卷: {ck}/{d}")
    volume.commit()
    print("[setup_models] DONE")


@app.function(
    gpu="L40S", volumes={VOL: volume}, timeout=1800,
)
def check_env():
    """复刻上次的环境自检 + Vulkan 状态检查。"""
    restore_checkpoint()
    os.chdir(SONIC)
    r = subprocess.run(
        "python check_environment.py --training", shell=True, capture_output=True, text=True
    )
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    v = subprocess.run(
        "vulkaninfo --summary 2>&1 | grep -E 'deviceName|driverName|apiVersion' || true",
        shell=True, capture_output=True, text=True,
    )
    print("=== Vulkan ===\n", v.stdout)


@app.function(volumes={VOL: volume}, timeout=600)
def convert(name: str = "01_running_man", out_name: str = "running_man_01"):
    """CSV → SONIC motion_lib PKL。"""
    restore_checkpoint()  # 转换脚本可能依赖仓库内配置
    os.chdir(SONIC)
    out = f"{VOL}/pkl/{out_name}.pkl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(
        f"python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py"
        f" --input {VOL}/motions/{name} --output {out} --fps 50",
        shell=True, check=True,
    )
    volume.commit()
    import joblib
    data = joblib.load(out)
    keys = list(data.keys()) if isinstance(data, dict) else type(data)
    print("[convert] keys:", keys)
    for k in ("fps", "pose_aa", "dof", "root_rot", "root_trans_offset"):
        if isinstance(data, dict) and k in data:
            print(f"  {k}: {getattr(data[k], 'shape', data[k])}")
    print("[convert] DONE ->", out)


@app.function(image=None, timeout=600)
def diag():
    """诊断 IsaacLab 安装状态。"""
    import subprocess
    r = subprocess.run("pip list 2>/dev/null | grep -i isaac", shell=True,
                       capture_output=True, text=True)
    print("pip 中的 isaac 包:\n", r.stdout)
    for mod in ("isaaclab", "isaaclab_tasks"):
        r2 = subprocess.run(
            f"python -c 'import {mod}; print({mod}.__file__)'",
            shell=True, capture_output=True, text=True)
        print(f"import {mod}: rc={r2.returncode}",
              (r2.stdout or r2.stderr).strip()[-600:])
    r3 = subprocess.run("ls /opt/IsaacLab/source", shell=True,
                        capture_output=True, text=True)
    print("IsaacLab/source:", r3.stdout)


# ---------------- 训练 ----------------
import shutil
import threading
import time

DATA_PKL = "zest_cartwheel_full.pkl"


def _backup_logs(local: str, vol: str, last_state: dict) -> None:
    if not os.path.isdir(local):
        return
    for root, _dirs, files in os.walk(local):
        rel = os.path.relpath(root, local)
        dst_root = os.path.join(vol, rel) if rel != "." else vol
        os.makedirs(dst_root, exist_ok=True)
        for fn in files:
            p = os.path.join(root, fn)
            mtime = os.path.getmtime(p)
            if last_state.get(p) == mtime:
                continue
            if fn.endswith((".pt", ".yaml", ".json", ".txt")) or mtime > time.time() - 3600:
                shutil.copy2(p, os.path.join(dst_root, fn))
                last_state[p] = mtime


@app.function(
    gpu="H200",
    volumes={VOL: volume},
    secrets=[hf_secret],
    timeout=10 * 3600,
)
def train_v1(num_iters: int = 4000, num_envs: int = 2048, lr: float = -1.0,
             anchor_w: float = -1.0, anchor_std: float = -1.0,
             ckpt: str = "sonic_release/last.pt", tag: str = "v1",
             motion: str = "zest_cartwheel_core",
             relax_terminations: bool = True) -> str:
    """V1 基础微调；传 lr/anchor_* 即 V1.1 风格 root refinement；ckpt 可换起点。

    relax_terminations=True（默认）: 把 tracking 终止阈值放宽到 100（等效禁用），
    让 episode 覆盖完整动作 —— 对应本项目的 continuation 配置。
    relax_terminations=False: 保留 exp 原始 strict 终止组（历史 Stage 1 口径）。
    """
    restore_checkpoint()
    if ckpt.startswith("/"):
        # 卷内路径：整个实验目录拷入（config.yaml 需与 ckpt 同目录）
        shutil.rmtree(f"{SONIC}/logs_rl/refine_start", ignore_errors=True)
        shutil.copytree(os.path.dirname(ckpt.rstrip("/")), f"{SONIC}/logs_rl/refine_start")
        ckpt = "logs_rl/refine_start/last.pt"
    os.makedirs(f"{SONIC}/data", exist_ok=True)
    src = f"{VOL}/pkl/{motion}.pkl"
    assert os.path.exists(src), f"缺少 {src}"
    shutil.copy2(src, f"{SONIC}/data/{motion}.pkl")

    os.chdir(SONIC)
    cmd = (
        "python gear_sonic/train_agent_trl.py"
        " +exp=manager/universal_token/all_modes/sonic_release"
        f" +checkpoint={ckpt}"
        f" num_envs={num_envs} headless=True"
        f" ++algo.config.num_learning_iterations={num_iters}"
        f" ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/{motion}.pkl"
        " ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros"
        " ++use_wandb=false"
    )
    if relax_terminations:
        cmd += (
            " ++manager_env.terminations.anchor_pos.params.threshold=100"
            " ++manager_env.terminations.anchor_pos.params.threshold_adaptive=false"
            " ++manager_env.terminations.ee_body_pos.params.threshold=100"
            " ++manager_env.terminations.ee_body_pos.params.threshold_adaptive=false"
            " ++manager_env.terminations.anchor_ori_full.params.threshold=100"
            " ++manager_env.terminations.foot_pos_xyz.params.threshold=100"
        )
    if lr > 0:
        cmd += f" ++algo.config.actor_learning_rate={lr}"
    if anchor_w > 0:
        cmd += f" ++manager_env.rewards.tracking_anchor_pos.weight={anchor_w}"
    if anchor_std > 0:
        cmd += f" ++manager_env.rewards.tracking_anchor_pos.params.std={anchor_std}"
    print("[train] CMD:", cmd, flush=True)

    logs_local = f"{SONIC}/logs_rl"
    logs_vol = f"{VOL}/logs_rl/{tag}"
    os.makedirs(logs_vol, exist_ok=True)

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def backup_loop() -> None:
        last: dict = {}
        while proc.poll() is None:
            time.sleep(300)
            try:
                _backup_logs(logs_local, logs_vol, last)
                volume.commit()
            except Exception as e:
                print("[backup] WARN:", e, flush=True)

    threading.Thread(target=backup_loop, daemon=True).start()
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    _backup_logs(logs_local, logs_vol, {})
    volume.commit()
    print(f"[train] exit={proc.returncode}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"训练失败 exit={proc.returncode}")

    runs = sorted(
        (d for d in os.listdir(logs_local) if os.path.isdir(f"{logs_local}/{d}")
         and not d.startswith(".")),
        key=lambda d: os.path.getmtime(f"{logs_local}/{d}"),
    )
    run_dir = f"{logs_vol}/{runs[-1]}" if runs else "?"
    print("[train] RUN DIR (volume):", run_dir, flush=True)
    return run_dir


@app.function(gpu="L40S", volumes={VOL: volume}, timeout=3600)
def run_eval(tag: str, ckpt: str = "sonic_release/last.pt"):
    """评估：patch im_eval 回调 dump npz -> 跑 eval -> 看门狗取 npz 存卷。

    ckpt: 相对 /opt/sonic 的路径，或卷内绝对路径（自动拷入）。
    """
    import time as _time
    restore_checkpoint()
    os.chdir(SONIC)
    if ckpt.startswith("/"):
        # 卷内路径：整个实验目录拷入（eval 需要 config.yaml/.hydra 与 ckpt 同目录）
        shutil.rmtree("/opt/sonic/eval_run", ignore_errors=True)
        shutil.copytree(os.path.dirname(ckpt.rstrip("/")), "/opt/sonic/eval_run")
        local_ck = "/opt/sonic/eval_run/last.pt"
        ckpt_rel = "eval_run/last.pt"
    else:
        local_ck = f"{SONIC}/{ckpt}"
        ckpt_rel = ckpt
    assert os.path.exists(local_ck), f"checkpoint 不存在: {local_ck}"
    os.makedirs(f"{SONIC}/data", exist_ok=True)
    shutil.copy2(f"{VOL}/pkl/{DATA_PKL}", f"{SONIC}/data/{DATA_PKL}")

    cb = f"{SONIC}/gear_sonic/trl/callbacks/im_eval_callback.py"
    code = open(cb, encoding="utf-8").read()
    if "[dump]" not in code:
        import re as _re
        lines = [
            "try:",
            "    import numpy as _np",
            "    _tonp = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else _np.asarray(x)",
            "    _np.savez('/tmp/eval_track_data.npz',",
            "              pred_pos_all=_np.array([_tonp(p) for p in self.pred_pos_all], dtype=object),",
            "              gt_pos_all=_np.array([_tonp(g) for g in self.gt_pos_all], dtype=object),",
            "              motion_keys=_np.array([_tonp(m) for m in self.sampled_motion_idx], dtype=object))",
            "    print('[dump] npz -> /tmp/eval_track_data.npz', flush=True)",
            "except Exception as _e:",
            "    print('[dump] failed:', _e, flush=True)",
        ]
        m = _re.search(r"^([ ]+)metrics_eval = self\._post_evaluate_policy\(actor_state\)", code, _re.M)
        assert m, "anchor not found"
        ind = m.group(1)
        inject = chr(10).join(ind + l for l in lines) + chr(10)
        code = code[:m.start()] + inject + code[m.start():]
        open(cb, "w", encoding="utf-8").write(code)
        print("[eval] im_eval_callback.py patched")
        r = subprocess.run(
            "python -c 'import gear_sonic.trl.callbacks.im_eval_callback' 2>&1 | tail -5",
            shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print("[eval] 导入失败!", r.stdout, r.stderr)
            r2 = subprocess.run(
                "grep -n -B2 -A14 '\[dump\]' /opt/sonic/gear_sonic/trl/callbacks/im_eval_callback.py | head -25",
                shell=True, capture_output=True, text=True)
            print(r2.stdout)
            raise RuntimeError("补丁破坏了导入")
    else:
        print("[eval] patch already present")

    cmd = (
        "HYDRA_FULL_ERROR=1 python gear_sonic/eval_agent_trl.py"
        f" +checkpoint={ckpt_rel} +headless=True"
        " ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=32"
        " '+manager_env/terminations=tracking/eval'"
        " ++manager_env.terminations.anchor_pos.params.threshold=100"
        " ++manager_env.terminations.anchor_ori_full.params.threshold=100"
        " ++manager_env.terminations.ee_body_pos.params.threshold=100"
        " ++manager_env.terminations.foot_pos_xyz.params.threshold=100"
        f" ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/{DATA_PKL}"
        " ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros"
        " ++use_wandb=false"
    )
    print("[eval] CMD:", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=lambda: [print(l, end="", flush=True) for l in proc.stdout],
                     daemon=True).start()

    npz_local = "/tmp/eval_track_data.npz"
    npz_vol = f"{VOL}/eval/eval_{tag}.npz"
    os.makedirs(f"{VOL}/eval", exist_ok=True)
    got = False
    deadline = _time.time() + 3000
    while _time.time() < deadline and proc.poll() is None:
        _time.sleep(10)
        if os.path.exists(npz_local) and _time.time() - os.path.getmtime(npz_local) > 20:
            got = True
            break
    if not got and os.path.exists(npz_local):
        got = True
    if got:
        shutil.copy2(npz_local, npz_vol)
        volume.commit()
        print("[eval] npz saved; terminating eval process (bypass metrics hang)", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    else:
        proc.kill()
        raise RuntimeError("eval did not produce npz")
    print(f"[eval] DONE -> {npz_vol}", flush=True)


@app.function(volumes={VOL: volume}, timeout=600)
def analyze_eval(tag: str, root_body: int = 0):
    """mpjpe_g / mpjpe_l / root drift（mm），与上次口径一致。"""
    import numpy as np
    d = np.load(f"{VOL}/eval/eval_{tag}.npz", allow_pickle=True)
    pred = list(d["pred_pos_all"])
    gt = list(d["gt_pos_all"])
    print(f"[analyze:{tag}] steps={len(pred)}")

    def to_arr(x):
        a = np.asarray(x, dtype=np.float64)
        return a[None] if a.ndim == 2 else a

    gs, ls, roots = [], [], []
    for p, g in zip(pred, gt):
        P, G = to_arr(p), to_arr(g)  # (N, B, 3)
        n = min(P.shape[0], G.shape[0])
        P, G = P[:n], G[:n]
        gs.append(np.linalg.norm(P - G, axis=-1).mean(axis=-1))
        Pr, Gr = P - P[:, root_body:root_body + 1], G - G[:, root_body:root_body + 1]
        ls.append(np.linalg.norm(Pr - Gr, axis=-1).mean(axis=-1))
        roots.append(np.linalg.norm(P[:, root_body] - G[:, root_body], axis=-1))
    mpjpe_g = float(np.concatenate(gs).mean()) * 1000
    mpjpe_l = float(np.concatenate(ls).mean()) * 1000
    root_drift = float(np.concatenate(roots).mean()) * 1000
    print(f"[analyze:{tag}] mpjpe_g={mpjpe_g:.1f} mm | mpjpe_l={mpjpe_l:.1f} mm | root_drift={root_drift:.1f} mm")
    return mpjpe_g, mpjpe_l, root_drift


@app.function(timeout=600)
def sh(cmd: str):
    """容器内跑任意命令（调试用）。"""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-8000:])
    if r.stderr.strip():
        print("STDERR:", r.stderr[-2000:])
    return r.returncode


@app.function(volumes={VOL: volume}, timeout=1800)
def vol_sh(cmd: str):
    """容器内跑命令并把结果写入卷（下载 checkpoint / 整理数据用）。"""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-8000:])
    if r.stderr.strip():
        print("STDERR:", r.stderr[-2000:])
    volume.commit()
    print("[vol_sh] exit =", r.returncode)
    return r.returncode


GL_DEB = ("https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
          "libnvidia-gl-580_580.95.05-0ubuntu1_amd64.deb")




def _setup_vulkan() -> None:
    """Modal GPU 容器 Vulkan 修复（vk_diag 验证通过的配方）。"""
    r = subprocess.run(
        f"curl -sL -o /tmp/gl.deb {GL_DEB} && dpkg-deb -x /tmp/gl.deb /"
        " && ln -sf libGLX_nvidia.so.580.95.05 /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"
        " && ldconfig && ls /usr/share/vulkan/icd.d/nvidia_icd.json",
        shell=True, capture_output=True, text=True)
    assert r.returncode == 0 and "nvidia_icd" in r.stdout, f"GL 解包失败: {r.stderr[-300:]}"
    print("[vulkan] gl 解包 OK", flush=True)
    # 用 .run 的 gpucomp 覆盖宿主版（与 Ubuntu 上游字节一致，消除混版本 DEVICE_LOST）
    r2 = subprocess.run(
        "curl -sL -o /tmp/nv.run https://us.download.nvidia.com/XFree86/Linux-x86_64/580.95.05/"
        "NVIDIA-Linux-x86_64-580.95.05.run && cd /tmp && sh nv.run -x >/dev/null 2>&1"
        " && cp /tmp/NVIDIA-Linux-x86_64-580.95.05/libnvidia-gpucomp.so.580.95.05"
        " /usr/lib/x86_64-linux-gnu/ && ldconfig && md5sum /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.580.95.05",
        shell=True, capture_output=True, text=True)
    print("[vulkan] gpucomp:", (r2.stdout or r2.stderr[-150:]).strip(), flush=True)
    r3 = subprocess.run(
        "if [ ! -e /dev/nvidia0 ]; then DEV=$(ls /dev/nvidia[0-9]* | head -1);"
        " MAJ=$(printf '%d' 0x$(stat -c %t $DEV)); MIN=$(printf '%d' 0x$(stat -c %T $DEV));"
        " mknod /dev/nvidia0 c $MAJ $MIN; fi; ls -l /dev/nvidia0 | head -1",
        shell=True, capture_output=True, text=True)
    print("[vulkan] nvidia0:", r3.stdout.strip(), flush=True)
    v = subprocess.run(
        "vulkaninfo --summary 2>&1 | grep -E 'deviceName' | head -3",
        shell=True, capture_output=True, text=True)
    print("[vulkan]", v.stdout.strip(), flush=True)
    assert ("L40S" in v.stdout or "A100" in v.stdout or "NVIDIA" in v.stdout), "NVIDIA Vulkan 未加载"


@app.function(gpu="H200", volumes={VOL: volume}, timeout=7200)
def render(tag: str, ckpt: str, offset: str = "[-6.5, 0, 0.55]", max_steps: int = 353):
    """渲染单个 checkpoint 的完整动作视频（正面固定机位）。"""
    _setup_vulkan()
    restore_checkpoint()
    os.makedirs(f"{SONIC}/data", exist_ok=True)
    shutil.copy2(f"{VOL}/pkl/{DATA_PKL}", f"{SONIC}/data/{DATA_PKL}")
    if ckpt.startswith("/"):
        shutil.rmtree(f"{SONIC}/logs_rl/render_ckpt", ignore_errors=True)
        shutil.copytree(os.path.dirname(ckpt.rstrip("/")), f"{SONIC}/logs_rl/render_ckpt")
        ckpt_rel = "logs_rl/render_ckpt/last.pt"
    else:
        ckpt_rel = ckpt

    out_dir = f"{VOL}/videos/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        "HYDRA_FULL_ERROR=1 python gear_sonic/eval_agent_trl.py"
        f" +checkpoint={ckpt_rel} +headless=True ++num_envs=1"
        " manager_env/recorders=render"
        " ++eval_callbacks=[]"
        " ++manager_env.config.render_results=true"
        f" ++manager_env.config.save_rendering_dir=/tmp/render_{tag}"
        " ++manager_env.config.fix_camera_after_first_frame=true"
        f" ++manager_env.config.eval_camera_offset=\"{offset}\""
        " ++manager_env.config.max_render_envs=1"
        " ++manager_env.config.render_frame_skip=2"
        f" ++max_render_steps={max_steps}"
        f" ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/{DATA_PKL}"
        " ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=zeros"
        " ++use_wandb=false"
    )
    # PNG 帧补丁：跳过 imageio 缓冲（崩溃不丢帧），帧直接落盘
    rec = f"{SONIC}/gear_sonic/envs/manager_env/mdp/recorders.py"
    code = open(rec, encoding="utf-8").read()
    if "png_fallback" not in code:
        anchor = "self.video_writers[i].append_data(frame)"
        repl = (
            "cv2.imwrite(f'{self.save_dir}/{self.start_idx+i:06d}_f{self.frame_id:05d}.png', frame)  # png_fallback"
        )
        assert anchor in code
        open(rec, "w", encoding="utf-8").write(code.replace(anchor, repl, 1))
        print("[render] PNG 帧补丁已应用", flush=True)

    print("[render] CMD:", cmd, flush=True)
    os.chdir(SONIC)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=lambda: [print(l, end="", flush=True) for l in proc.stdout],
                     daemon=True).start()
    rc = proc.wait()
    print(f"[render] exit={rc}", flush=True)
    src_dir = f"/tmp/render_{tag}"
    n = 0
    if os.path.isdir(src_dir):
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith(".mp4"):
                shutil.copy2(f"{src_dir}/{fn}", f"{out_dir}/{fn}")
                n += 1
    # PNG 帧 -> mp4（崩溃也能合成已有帧）
    import glob as _glob
    pngs = sorted(_glob.glob(f"{src_dir}/*_f*.png"))
    if pngs:
        import cv2  # noqa
        first = cv2.imread(pngs[0])
        h, w = first.shape[:2]
        mp4_path = f"{out_dir}/{tag}.mp4"
        vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"avc1"), 25, (w, h))
        for p in pngs:
            vw.write(cv2.imread(p))
        vw.release()
        print(f"[render] PNG 合成: {len(pngs)} 帧 -> {mp4_path}", flush=True)
        n = max(n, 1)
    volume.commit()
    print(f"[render] {n} 个视频 -> {out_dir}", flush=True)
    if rc != 0 and n == 0:
        raise RuntimeError(f"渲染失败 exit={rc}")


@app.function(gpu="L40S", volumes={VOL: volume}, timeout=3600)
def export_onnx(tag: str, ckpt: str):
    """导出 ONNX（提交产物）。"""
    restore_checkpoint()
    os.makedirs(f"{SONIC}/data", exist_ok=True)
    shutil.copy2(f"{VOL}/pkl/{DATA_PKL}", f"{SONIC}/data/{DATA_PKL}")
    if ckpt.startswith("/"):
        shutil.rmtree(f"{SONIC}/logs_rl/onnx_ckpt", ignore_errors=True)
        shutil.copytree(os.path.dirname(ckpt.rstrip("/")), f"{SONIC}/logs_rl/onnx_ckpt")
        ckpt_rel = "logs_rl/onnx_ckpt/last.pt"
    else:
        ckpt_rel = ckpt
    os.chdir(SONIC)
    cmd = (
        "HYDRA_FULL_ERROR=1 python gear_sonic/eval_agent_trl.py"
        f" +checkpoint={ckpt_rel} +headless=True ++num_envs=1"
        " +export_onnx_only=true ++use_wandb=false"
    )
    print("[onnx] CMD:", cmd, flush=True)
    open("/tmp/onnx_marker", "w").close()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    print(f"[onnx] exit={rc}", flush=True)
    out_dir = f"{VOL}/onnx/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    import subprocess as _sp
    found = _sp.run(
        f"find {SONIC} -name '*.onnx' -newer /tmp/onnx_marker 2>/dev/null",
        shell=True, capture_output=True, text=True).stdout.split()
    n = 0
    for p in found:
        shutil.copy2(p, f"{out_dir}/{os.path.basename(p)}")
        print("[onnx] +", os.path.basename(p), flush=True)
        n += 1
    volume.commit()
    print(f"[onnx] {n} 个文件 -> {out_dir}", flush=True)
    if rc != 0:
        raise RuntimeError(f"ONNX 导出失败 exit={rc}")


@app.function(volumes={VOL: volume}, timeout=3600, cpu=4)
def render_skeleton(tag: str, eval_tag: str, title: str = "", azim: int = 90):
    """从评估 npz 的 rollout 轨迹生成骨架动画视频（CPU，无需 Vulkan）。"""
    import glob as _glob
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    d = np.load(f"{VOL}/eval/eval_{eval_tag}.npz", allow_pickle=True)
    pred = [np.asarray(p, dtype=np.float64) for p in d["pred_pos_all"]]
    gt = [np.asarray(g, dtype=np.float64) for g in d["gt_pos_all"]]
    # 每个条目是一条 rollout (T, B, 3)：选平均逐帧误差最小的 rollout
    errs = [np.mean([np.linalg.norm(pred[e][t] - gt[e][t], axis=-1).mean()
                     for t in range(min(len(pred[e]), len(gt[e])))])
            for e in range(len(pred))]
    best = int(np.argmin(errs))
    P, G = pred[best], gt[best]
    T = min(len(P), len(G))
    P, G = P[:T], G[:T]
    B = P.shape[1]
    print(f"[skel:{tag}] T={T} B={B} rollout={best} err={errs[best]*1000:.0f}mm", flush=True)

    # parent 连线（跳过 world=0）
    edges = [(i, p_) for i, p_ in enumerate(
        [0, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 16]
        + [25, 26, 27, 28, 29][: B - 25]) if i >= 1 and p_ >= 1]
    # 上面硬编码对不上就退化为: 所有 body 与最近邻连线
    if len(edges) < 10 or max(max(e) for e in edges) >= B:
        edges = []
        for i in range(1, B):
            dists = np.linalg.norm(P[:, i:i + 1] - P[:, 1:], axis=2).mean(0)
            dists[i - 1] = 1e9
            edges.append((i, int(np.argmin(dists)) + 1))

    lo = np.minimum(P.reshape(-1, 3), G.reshape(-1, 3)).min(0) - 0.3
    hi = np.maximum(P.reshape(-1, 3), G.reshape(-1, 3)).max(0) + 0.3
    ctr = (lo + hi) / 2
    rng = (hi - lo).max() / 2

    out_dir = f"{VOL}/videos/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    mp4 = f"{out_dir}/{tag}_skeleton.mp4"

    import cv2
    step = 2  # 50Hz -> 25fps
    frames = range(0, T, step)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    try:
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"avc1"), 25,
                             (1280, 720))
        if not vw.isOpened():
            vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
        for t in frames:
            fig.clf()
            for k, (data, name, color) in enumerate(
                    ((P[t], f"Policy: {title or tag}", "tab:blue"),
                     (G[t], "Reference motion", "tab:gray"))):
                ax = fig.add_subplot(1, 2, k + 1, projection="3d")
                for i, j in edges:
                    ax.plot(*zip(data[i], data[j]), color=color, linewidth=2.5)
                ax.scatter(*data.T, s=8, c=[color])
                ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
                ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
                ax.set_zspan = None
                ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
                ax.view_init(elev=12, azim=azim)
                ax.set_title(name)
                ax.set_box_aspect((1, 1, 1))
                ax.set_axis_off()
            fig.suptitle(f"{title or tag}  |  rollout step {t+1}/{T}", fontsize=14)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            vw.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        vw.release()
    finally:
        plt.close(fig)
    volume.commit()
    sz = os.path.getsize(mp4)
    print(f"[skel:{tag}] DONE {mp4} ({sz/1e6:.1f} MB, {len(list(frames))} 帧)", flush=True)
