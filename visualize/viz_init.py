# viz_init.py
from __future__ import annotations
from manim import *
import numpy as np, argparse, os
from pathlib import Path
from viz_common import (
    NetworkVizBase, load_meta, load_epoch, load_sample_image,
)

class MLPInitScene(NetworkVizBase):
    """
    初始化：展示网络骨架、载入图片、像素→向量条动画，并加载“初始网络”的权重连线。
    不展示训练/反向/切换。
    """
    def __init__(self, log_dir: str | None = None, epoch: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._log_dir = log_dir or os.environ.get("MLP_LOG_DIR", "")
        self._epoch = epoch

    def construct(self):
        assert self._log_dir, "log_dir 未指定（--log-dir 或环境变量 MLP_LOG_DIR）"
        meta = load_meta(self._log_dir)
        start_epoch = int(self._epoch or 1)
        ep = load_epoch(self._log_dir, start_epoch)

        # 参数
        self.input_size = int(meta.get("input_size", 784))
        self.hidden_size = int(meta.get("hidden_size", 8))
        self.output_size = int(meta.get("output_size", 10))
        self.learning_rate = float(meta.get("learning_rate", 0.1))
        self.topk_w1 = int(meta.get("topk_w1", 50))

        # 图像
        img = load_sample_image(self._log_dir)  # 28x28, [0,1]
        base_img, border, bar_container, targets = self._build_input_bar(img)

        # 网络骨架
        self.targets = targets
        self.neurons, self.out_neurons = self._build_network_nodes(bar_container)

        # 标题：Epoch
        total = int(meta.get("epochs", start_epoch))
        self.show_epoch_title(start_epoch, total)

        # 初始网络（使用该 epoch 的 ori 权重）
        self.weight_1, self.idx_in_per_hidden = self.build_lines_W1(ep["oriW1"])
        self.weight_2 = self.build_lines_W2(ep["oriW2"])

        # 结束停顿
        self.wait(1.2)

def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epoch", type=int, default=1, help="初始化所用的日志 epoch（默认 1，对应 ori 权重）")
    ap.add_argument("--quality", default="high_quality",
                    choices=["low_quality","medium_quality","high_quality","fourk_quality"])
    ap.add_argument("--out", default=None, help="输出文件名（不含扩展名）")
    ap.add_argument("--outdir", default=None, help="最终视频输出文件夹")
    ap.add_argument("--tmpdir", default=None, help="中间文件根目录（并行时用于隔离缓存）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    os.environ["MLP_LOG_DIR"] = args.log_dir
    from manim import config
    config.quality = args.quality
    if args.out and args.outdir:
        full_path = Path(args.outdir).resolve() / f"{args.out}.mp4"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_file = str(full_path)   # ✅ 只设输出文件
    elif args.out:
        config.output_file = args.out         # 回落：仅修改文件名

    if args.tmpdir:
        tmp = Path(args.tmpdir).resolve()
        tmp.mkdir(parents=True, exist_ok=True)
        config.media_dir = str(tmp)            # 所有缓存、Tex、partial 都放这里（每进程独立）
    scene = MLPInitScene(log_dir=args.log_dir, epoch=args.epoch)
    scene.render()
