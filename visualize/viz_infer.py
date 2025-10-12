# viz_infer.py
from __future__ import annotations
from manim import *
import numpy as np, argparse, os
from pathlib import Path
from PIL import Image
from viz_common import (
    NetworkVizBase, load_meta, load_sample_image, load_mnist_image
)

DEFAULT_TEST_UBYTE = "assets/MNIST/raw/t10k-images-idx3-ubyte"

def load_model_npz(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    data = np.load(model_path)
    return data["W1"], data["b1"], data["W2"], data["b2"]

def load_gray_image(image_path: str, target_size: int = 28, invert: bool = False) -> np.ndarray:
    """读取任意图片 -> 灰度 -> resize 28x28 -> 归一化到 [0,1]。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image file not found: {image_path}")
    img = Image.open(image_path).convert("L")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr  # (28,28), [0,1]

class MLPInferenceScene(NetworkVizBase):
    """
    推理验证：加载最终模型（或指定 checkpoint），并按优先级选择输入图片：
      1) --image-file：任意灰度图（自动转 28x28，[0,1]），可配 --invert
      2) --index：从测试集 ubyte 读取（默认路径 DEFAULT_TEST_UBYTE，或用 --test-ubyte 指定）
      3) 否则：从日志目录读取 sample_for_viz.npy（对应 meta.viz_image_index）
    """
    def __init__(self, log_dir: str,
                 model_path: str | None = None,
                 image_file: str | None = None,
                 invert: bool = False,
                 index: int | None = None,
                 test_ubyte: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._log_dir = log_dir
        self._model_path = model_path
        self._image_file = image_file
        self._invert = invert
        self._index = index
        self._test_ubyte = test_ubyte

    def construct(self):
        assert self._log_dir, "log_dir 未指定（--log-dir 或环境变量 MLP_LOG_DIR）"
        meta = load_meta(self._log_dir)

        # 同步网络参数（仅用于可视化 UI）
        self.input_size = int(meta.get("input_size", 784))
        self.hidden_size = int(meta.get("hidden_size", 8))
        self.output_size = int(meta.get("output_size", 10))
        self.learning_rate = float(meta.get("learning_rate", 0.1))
        self.topk_w1 = int(meta.get("topk_w1", 50))

        # 模型
        model_path = self._model_path or os.path.join(self._log_dir, "model_last.npz")
        W1, b1, W2, b2 = load_model_npz(model_path)

        # ===== 图片选择逻辑（满足你的三段优先级） =====
        if self._image_file:  # 1) 任意图片
            img = load_gray_image(self._image_file, target_size=28, invert=self._invert)
        elif self._index is not None:  # 2) 测试集索引
            test_path = self._test_ubyte or DEFAULT_TEST_UBYTE
            if not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"未找到默认测试集文件：{test_path}\n"
                    f"请通过 --test-ubyte 显式传入，例如：--test-ubyte assets/MNIST/raw/t10k-images-idx3-ubyte"
                )
            img = load_mnist_image(test_path, self._index)  # (28,28), [0,1]
        else:  # 3) 回退到训练时保存的 sample_for_viz.npy（对应 meta.viz_image_index）
            img = load_sample_image(self._log_dir)  # (28,28), [0,1]

        # UI
        base_img, border, bar_container, targets = self._build_input_bar(img)
        self.targets = targets
        self.neurons, self.out_neurons = self._build_network_nodes(bar_container)

        # 标题（推理无 epoch 概念）
        self.show_epoch_title(epoch=0, total=None)

        # 构建网络线（使用最终权重）
        self.weight_1, self.idx_in_per_hidden = self.build_lines_W1(W1)
        self.weight_2 = self.build_lines_W2(W2)

        # 前向可视
        x = img.flatten().astype(np.float64)
        out_values, Z2 = self.forward_and_visualize(x, W1, b1, W2, b2)

        # Top-1 预测显示
        pred = int(np.argmax(Z2))
        label_txt = Text(f"Prediction: {pred}", font_size=36, color=YELLOW)\
                        .next_to(self.out_neurons, DOWN, buff=0.8).align_to(self.out_neurons, LEFT)
        self.play(FadeIn(label_txt, shift=UP*0.2), run_time=0.35)
        self.wait(1.5)

def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="训练日志目录（包含 model_last.npz / sample_for_viz.npy / meta.json）")
    ap.add_argument("--model", default=None, help="模型 npz 路径，默认读取 log_dir/model_last.npz")
    # 通用图片输入
    ap.add_argument("--image-file", default=None, help="任意灰度图片路径（自动转 28x28 并归一化）")
    ap.add_argument("--invert", action="store_true", help="对 --image-file 的灰度进行反相（黑白互换）")
    # 测试集索引输入
    ap.add_argument("--index", type=int, default=None, help="测试集图片索引（从 t10k ubyte 读取）")
    ap.add_argument("--test-ubyte", default=None, help="测试集 ubyte 路径（默认 assets/MNIST/raw/t10k-images-idx3-ubyte）")
    # 输出/质量
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
        config.output_file = str(full_path)
    elif args.out:
        config.output_file = args.out

    if args.tmpdir:
        tmp = Path(args.tmpdir).resolve()
        tmp.mkdir(parents=True, exist_ok=True)
        config.media_dir = str(tmp)

    scene = MLPInferenceScene(
        log_dir=args.log_dir,
        model_path=args.model,
        image_file=args.image_file,
        invert=args.invert,
        index=args.index,
        test_ubyte=args.test_ubyte
    )
    scene.render()
