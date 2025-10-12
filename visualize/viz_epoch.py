# viz_epoch.py
from __future__ import annotations
from manim import *
import numpy as np, argparse, os
from pathlib import Path
from viz_common import (
    NetworkVizBase, load_meta, load_epoch, load_sample_image,
    # 用到的常量
    BAR_WIDTH, BAR_HEIGHT, PIXEL_GAP, RIGHT_GAP, LAYER_GAP,
    NEURON_R, OUT_NEURON_R, NEURON_SPACING, OUT_SPACING,
    gray_color,
)

class MLPMidEpochScene(NetworkVizBase):
    """
    中间 epoch 场景：
    - 开场即显示“已加载好的网络”（上一轮结束后的样子 == 本 epoch 的 ori 权重），无任何入场动画；
    - 不展示图片加载或像素→向量化过程（默认也不显示原始图片，可选 --show-input 仅静态展示，不做动画）；
    - 之后执行：前向贡献→loss 列表与平均→切回权重样式并压缩→反向→学习率→切换到 res 新权重。
    """
    def __init__(self, log_dir: str | None = None, epoch: int | None = None,
                 show_input: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._log_dir = log_dir or os.environ.get("MLP_LOG_DIR", "")
        self._epoch = epoch
        self._show_input = show_input

    # ---------------- 静态构建：无动画 ----------------
    def _make_static_input_bar(self, img_28x28: np.ndarray):
        """
        只构建输入条（targets）与容器，不显示原始图片、不做像素->条形动画。
        返回：(bar_container, targets)
        """
        # 在左侧放一个不可见的“占位”矩形，便于定位
        base_h = 2.8  # 保持与初始化场景相近的布局比例
        placeholder = Rectangle(width=base_h*1.0, height=base_h).set_stroke(width=0).to_edge(LEFT)
        self.add(placeholder)

        border = SurroundingRectangle(placeholder, buff=0.06, stroke_width=0, color=BLACK)  # 无边框
        self.add(border)

        # 构建输入条容器与 784 个条形（直接加入场景）
        bar_container = Rectangle(width=BAR_WIDTH, height=BAR_HEIGHT).set_stroke(WHITE, 2)
        bar_container.next_to(border, RIGHT, buff=1.0)
        self.add(bar_container)

        # 构建 targets（按图片灰度，但直接显示；无动画）
        bar_top = bar_container.get_top()[1]; bar_x = bar_container.get_center()[0]
        step = BAR_HEIGHT / self.input_size; cell_h = step * 0.95

        targets = VGroup()
        for i in range(self.input_size):
            cy = bar_top - (i + 0.5) * step
            r, c = divmod(i, 28)
            v = img_28x28[r, c]
            if v <= 1.0:
                v = int(v * 255)
            rect = Rectangle(width=BAR_WIDTH * 0.9, height=cell_h).set_stroke(width=0)
            rect.set_fill(gray_color(v), opacity=1.0).move_to([bar_x, cy, 0])
            targets.add(rect)
        self.add(targets)

        # 1) 原始图片（静态，统一边框样式）
        
        img_mobj, border = self._add_image_with_border(img_28x28)
        left_anchor = border
        
        return bar_container, targets

    def _make_static_nodes(self, bar_container):
        """静态构建隐藏层与输出层节点（不播放 FadeIn/FadeOut）。"""
        base_x = bar_container.get_right()[0] + RIGHT_GAP
        base_y = bar_container.get_center()[1]

        neurons = VGroup()
        for j in range(self.hidden_size):
            cy = base_y + ((7/2 - j) * NEURON_SPACING)
            circle = Circle(radius=NEURON_R).set_stroke(WHITE, 1.5).set_fill(YELLOW, opacity=0)
            circle.move_to([base_x, cy, 0])
            neurons.add(circle)
        self.add(neurons)

        out_x = base_x + LAYER_GAP
        out_y = base_y
        out_neurons = VGroup()
        for k in range(self.output_size):
            cy = out_y + ((9/2 - k) * OUT_SPACING)
            oc = Circle(radius=OUT_NEURON_R).set_stroke(WHITE, 1.5).set_fill(YELLOW, opacity=0)
            oc.move_to([out_x, cy, 0])
            out_neurons.add(oc)
        self.add(out_neurons)

        return neurons, out_neurons

    def _set_epoch_title_static(self, epoch: int, total: int):
        """不播放 Transform/ FadeIn，直接把 Epoch 文本加入场景。"""
        txt = Text(f"Epoch {epoch}/{total}", font_size=30, color=self.colors[-1])\
                .to_corner(UL).shift(DOWN*0.2 + RIGHT*0.2)
        self.add(txt)

    # ---------------- 主流程 ----------------
    def construct(self):
        assert self._log_dir, "log_dir 未指定（--log-dir 或环境变量 MLP_LOG_DIR）"
        meta = load_meta(self._log_dir)
        epoch = int(self._epoch or meta.get("epochs", 1))
        ep = load_epoch(self._log_dir, epoch)

        # 参数
        self.input_size  = int(meta.get("input_size", 784))
        self.hidden_size = int(meta.get("hidden_size", 8))
        self.output_size = int(meta.get("output_size", 10))
        self.learning_rate = float(meta.get("learning_rate", 0.1))
        self.topk_w1 = int(meta.get("topk_w1", 50))

        # 准备输入（仅用于数值与 targets 的填充；不播放任何“加载/变换”段落）
        img = load_sample_image(self._log_dir)  # 28x28 [0,1]
        bar_container, targets = self._make_static_input_bar(img)
        self.targets = targets
        self.neurons, self.out_neurons = self._make_static_nodes(bar_container)

        # 标题静态加入
        self._set_epoch_title_static(epoch, int(meta.get("epochs", epoch)))

        # —— 开场即“已加载好的网络”：直接把连线加入场景（无 Create/FadeIn） —— #
        self.weight_1, self.idx_in_per_hidden = self.build_lines_W1(ep["oriW1"], animated=False)
        self.weight_2 = self.build_lines_W2(ep["oriW2"], animated=False)

        # ===== 本轮可视化 =====
        # 前向贡献 & logits（基于“旧网络”）
        x = img.flatten().astype(np.float64)
        out_values, _ = self.forward_and_visualize(x, ep["oriW1"], ep["orib1"], ep["oriW2"], ep["orib2"])

        # loss 列表与平均
        losses = ep["losses"] if ep["losses"] is not None else np.array([], dtype=np.float32)
        loss_texts = self.show_loss_list(out_values, losses)
        loss_circle, aveloss_tex = self.show_avg_loss(loss_texts, losses)
        self.reg_epoch_objs(loss_circle, aveloss_tex)

        # 切回权重样式并压缩
        self.lines_to_weight_style(ep["oriW1"], ep["oriW2"])  # 第二个参数只用到 W2；安全起见传 oriW2
        self.compact_network()

        # 反向可视化（若日志缺失则自动跳过）
        conns = []
        if ep["avedZ2"] is not None:
            conns.append(self.backward_loss_to_output(loss_circle, ep["avedZ2"]))
        if ep["avedW2"] is not None:
            conns.append(self.backward_output_to_hidden(ep["avedW2"]))
        if ep["avedW1"] is not None:
            conns.append(self.backward_hidden_to_input_gradonly(ep["avedW1"]))
        if conns:
            self.show_lr_and_decay(*conns)
            self.reg_epoch_objs(*conns)

        # 切换到“新权重”（W2 参数更新；W1 重建）
        self.swap_to_new_weights(ep["resW1"], ep["resW2"])
        self.clear_epoch_bin()
        self.wait(0.6)

# ------- CLI -------
def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epoch", type=int, required=True, help="要渲染的 epoch（使用该 epoch 的 npz）")
    ap.add_argument("--quality", default="high_quality",
                    choices=["low_quality","medium_quality","high_quality","fourk_quality"])
    ap.add_argument("--show-input", action="store_true", help="静态显示原始图片（无动画）；默认不显示")
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
        config.media_dir = str(tmp)   

    scene = MLPMidEpochScene(
        log_dir=args.log_dir,
        epoch=args.epoch,
        show_input=args.show_input,
    )
    scene.render()
