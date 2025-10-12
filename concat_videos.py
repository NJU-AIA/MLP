#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path
import sys

def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def write_list_file(src_dir: Path, names: list[str]) -> Path:
    lst = src_dir / "concat_list.txt"
    with lst.open("w", encoding="utf-8") as f:
        for n in names:
            f.write(f"file '{(src_dir / (n + '.mp4')).as_posix()}'\n")
    return lst

def concat_copy(src_dir: Path, out_file: Path, names: list[str]) -> bool:
    lst = write_list_file(src_dir, names)
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_file)]
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0

def concat_reencode(src_dir: Path, out_file: Path, names: list[str]) -> bool:
    lst = write_list_file(src_dir, names)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(out_file),
    ]
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0

def main():
    ap = argparse.ArgumentParser(description="Concat init + epochs + infer into one video.")
    ap.add_argument("--src", required=True, help="片段所在目录（包含 init/epoch_xxx/infer）")
    ap.add_argument("--epochs", type=int, default=8, help="epoch 数量")
    ap.add_argument("--out", default=None, help="输出文件路径（默认为 <src>/full.mp4）")
    ap.add_argument("--force-reencode", action="store_true", help="直接重编码（跳过无重编码）")
    args = ap.parse_args()

    src_dir = Path(args.src).resolve()
    if not src_dir.exists():
        print(f"[ERROR] 目录不存在: {src_dir}")
        sys.exit(2)

    order = ["init"] + [f"epoch_{i:03d}" for i in range(1, args.epochs + 1)] + ["infer"]
    # 校验片段存在
    missing = [name for name in order if not (src_dir / f"{name}.mp4").exists()]
    if missing:
        print("[ERROR] 缺少片段：", ", ".join(missing))
        print("请检查并确保以上 mp4 文件都已生成。")
        sys.exit(3)

    if not have_ffmpeg():
        print("[ERROR] 未找到 ffmpeg，请安装后加入 PATH。")
        sys.exit(4)

    out_file = Path(args.out).resolve() if args.out else (src_dir / "full.mp4")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 输出文件: {out_file}")

    ok = False
    ok = concat_reencode(src_dir, out_file, order)
    

    if ok:
        print(f"[OK] 拼接完成：{out_file}")
    else:
        print("[ERROR] 拼接失败。")
        sys.exit(5)

if __name__ == "__main__":
    main()
