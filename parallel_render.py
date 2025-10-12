#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import concurrent.futures
import os
import sys
from pathlib import Path
import subprocess

PYTHON = sys.executable
LOG_DIR = "runs/mlp1"
QUALITY = "medium_quality"
OUTDIR = Path("renders/mlp1")        # 最终成片目录
TMPROOT = Path(".manim_tmp_parallel")  # 中间文件根目录（每任务一个子目录）
EPOCHS = 8

OUTDIR.mkdir(parents=True, exist_ok=True)
TMPROOT.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, env=None):
    print(">>", " ".join(map(str, cmd)))
    return subprocess.run(cmd, env=env).returncode

def render_init():
    tmp = TMPROOT / "init"
    return run_cmd([
        PYTHON, "visualize/viz_init.py",
        "--log-dir", LOG_DIR,
        "--quality", QUALITY,
        "--out", "init",
        "--outdir", str(OUTDIR),
        "--tmpdir", str(tmp),
    ])

def render_epoch(i: int):
    tmp = TMPROOT / f"epoch_{i:03d}"
    return run_cmd([
        PYTHON, "visualize/viz_epoch.py",
        "--log-dir", LOG_DIR,
        "--epoch", str(i),
        "--quality", QUALITY,
        "--out", f"epoch_{i:03d}",
        "--outdir", str(OUTDIR),
        "--tmpdir", str(tmp),
    ])

def render_infer():
    tmp = TMPROOT / "infer"
    return run_cmd([
        PYTHON, "visualize/viz_infer.py",
        "--log-dir", LOG_DIR,
        "--quality", QUALITY,
        "--out", "infer",
        "--outdir", str(OUTDIR),
        "--tmpdir", str(tmp),
        "--index", "68",
    ])

def main():
    jobs = []
    max_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        jobs.append(ex.submit(render_init))
        jobs.append(ex.submit(render_infer))
        for i in range(1, EPOCHS + 1):
            jobs.append(ex.submit(render_epoch, i))

        ok = True
        for fut in concurrent.futures.as_completed(jobs):
            rc = fut.result()
            if rc != 0:
                ok = False
                print(f"[WARN] 子任务返回码 {rc}")

    print(f"\n[{'OK' if ok else 'DONE'}] 渲染结束。视频在: {OUTDIR}")

if __name__ == "__main__":
    main()
