# viz_common.py
from __future__ import annotations
from manim import *
import numpy as np
import json, os
from PIL import Image

# ========= 基础数值函数 =========
def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    """数值稳定版 softmax，输入 1D 向量，返回概率"""
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)          # 防止溢出
    e = np.exp(z)
    s = e / np.sum(e)
    return s

def norm01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def gray_color(v: float):
    # v: 0~255
    return interpolate_color(BLACK, WHITE, float(v)/255.0)

# ========= 日志/模型加载 =========
def load_meta(log_dir: str) -> dict:
    with open(os.path.join(log_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_epoch(log_dir: str, epoch: int) -> dict[str, np.ndarray | int]:
    path = os.path.join(log_dir, f"epoch_{epoch:03d}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"epoch file not found: {path}")
    ep = np.load(path)
    return {
        "oriW1": ep["oriW1"], "orib1": ep["orib1"],
        "oriW2": ep["oriW2"], "orib2": ep["orib2"],
        "resW1": ep["resW1"], "resb1": ep["resb1"],
        "resW2": ep["resW2"], "resb2": ep["resb2"],
        "avedW1": ep.get("avedW1", None), "avedb1": ep.get("avedb1", None),
        "avedW2": ep.get("avedW2", None), "avedb2": ep.get("avedb2", None),
        "avedZ2": ep.get("avedZ2", None), "losses": ep.get("losses", None),
    }

def load_sample_image(log_dir: str) -> np.ndarray:
    # 28x28, 值域[0,1]
    p = os.path.join(log_dir, "sample_for_viz.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"sample_for_viz.npy not found in {log_dir}")
    return np.load(p)

def load_mnist_image(ubyte_path: str, index: int) -> np.ndarray:
    # 返回 28x28, 值域[0,1]
    with open(ubyte_path, 'rb') as f:
        magic, size = np.frombuffer(f.read(8), dtype='>i4')
        rows, cols = np.frombuffer(f.read(8), dtype='>i4')
        if index < 0 or index >= size:
            raise IndexError(f"index out of range: 0..{size-1}")
        f.seek(16 + index*rows*cols)
        buf = f.read(rows*cols)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols)
    return img.astype(np.float32)/255.0

# ========= 常量 =========
IMG_HEIGHT = 2.8
BORDER_BUFF = 0.06
BORDER_STROKE_WIDTH = 2.5
BORDER_COLOR = WHITE
BAR_HEIGHT = 4.6
BAR_WIDTH  = 0.6
PIXEL_GAP  = 0.03
ANIM_TIME  = 2.6
LAG_RATIO  = 0.02

NEURON_R   = 0.22
NEURON_SPACING = 0.55
RIGHT_GAP = 1.8

OUT_NEURON_R = 0.22
OUT_SPACING  = 0.48
LAYER_GAP    = 2.4

CURVE_BEND = 0.15
WEIGHT_WIDTH_MAX_1  = 3.0
WEIGHT_OPAC_MAX_1   = 0.75
CONTRIB_WIDTH_MAX_1 = 3.0
CONTRIB_OPAC_MAX_1  = 0.85

WEIGHT_WIDTH_MAX_2  = 3.0
WEIGHT_OPAC_MAX_2   = 0.75
CONTRIB_WIDTH_MAX_2 = 3.0
CONTRIB_OPAC_MAX_2  = 0.85

LOSS_BUFF = 1.6
LOSS_FONT_SIZE = 24
LOSS_MAX_DISPLAY = 14
LAG_RATIO_LOSS = 0.05
LOSS_SPACING = 0.48
LOSS_R = 0.7
AVELOSS_FONT_SIZE = 36

DZ2_WIDTH_MAX = 3.0
DZ2_OPAC_MAX = 0.75

DW2_WIDTH_MAX  = 3.0
DW2_OPAC_MAX   = 0.75

DW1_WIDTH_MAX  = 3.0
DW1_OPAC_MAX   = 0.4

LEARNING_RATE_FONT_SIZE = 36

COMPACT_SHIFT = UP*3.35
COMPACT_SCALE = 0.25

W2_WIDTH_MAX  = 3.0
W2_OPAC_MAX   = 0.75
W1_WIDTH_MAX  = 3.0
W1_OPAC_MAX   = 0.4

# ========= 基类：封装公共绘制/动画 =========
class NetworkVizBase(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = 784
        self.hidden_size = 8
        self.output_size = 10
        self.learning_rate = 0.1
        self.topk_w1 = 50
        self.FAST_MODE = bool(int(os.getenv("FAST_MODE", "0")))
        # 颜色带
        self.colors = [interpolate_color(WHITE, GRAY, t) for t in np.linspace(0, 0.8, 16)]
        self._init_epoch_bin()

    # ----- UI -----
    def _build_input_bar(self, img_uint8_28x28: np.ndarray):
        # to RGB & disable cache to avoid texture reuse
        rgb = np.stack([img_uint8_28x28]*3, axis=-1).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb).copy()
        pil_img = Image.fromarray(rgb, mode="RGB")
        with tempconfig({"disable_caching": True}):
            base_img = ImageMobject(pil_img)

        base_img.set_height(IMG_HEIGHT).to_edge(LEFT)
        border = SurroundingRectangle(base_img, buff=0.06, stroke_width=2.5, color=WHITE)
        self.play(FadeIn(base_img), Create(border), run_time=0.6)

        # 👉 为了“抽空效果”，先把左侧整张原图暂时隐藏（像素移动时左侧会变空）
        self.play(base_img.animate.set_opacity(0.0), run_time=0.15)

        # pixels
        left, right = base_img.get_left()[0], base_img.get_right()[0]
        bottom, top = base_img.get_bottom()[1], base_img.get_top()[1]
        dx, dy = (right-left)/28, (top-bottom)/28
        side = min(dx, dy)*(1-PIXEL_GAP)

        pixel_rects = VGroup()
        for r in range(28):
            for c in range(28):
                cx = left + (c+0.5)*dx; cy = top - (r+0.5)*dy
                rect = Rectangle(width=side, height=side).set_stroke(width=0)
                rect.set_fill(
                    gray_color(int(img_uint8_28x28[r, c]*255) if img_uint8_28x28.max()<=1.0 else img_uint8_28x28[r,c]),
                    opacity=1.0
                ).move_to([cx, cy, 0])
                pixel_rects.add(rect)
        self.play(FadeIn(pixel_rects, lag_ratio=0.01), run_time=0.4)

        # bar container + targets
        bar_container = Rectangle(width=BAR_WIDTH, height=BAR_HEIGHT).set_stroke(WHITE, 2)
        bar_container.next_to(border, RIGHT, buff=1.0)
        self.play(Create(bar_container), run_time=0.3)

        bar_top = bar_container.get_top()[1]; bar_x = bar_container.get_center()[0]
        step = BAR_HEIGHT/self.input_size; cell_h = step*0.95

        targets = VGroup()
        for i in range(self.input_size):
            cy = bar_top - (i+0.5)*step; r, c = divmod(i, 28)
            v = img_uint8_28x28[r, c]
            v = int(v*255) if v<=1.0 else int(v)
            rect = Rectangle(width=BAR_WIDTH*0.9, height=cell_h).set_stroke(width=0)
            rect.set_fill(gray_color(v), opacity=1.0).move_to([bar_x, cy, 0])
            targets.add(rect)

        
        transforms = [Transform(src, dst, replace_mobject_with_target_in_scene=True)
                    for src, dst in zip(pixel_rects, targets)]
        self.play(AnimationGroup(*transforms, lag_ratio=0.02), run_time=2.6)
        self.play(FadeOut(pixel_rects, shift=RIGHT*0.2), run_time=0.3)
        # 👉 回显原图
        self._add_image_with_border(img_uint8_28x28)

        return base_img, border, bar_container, targets

    
    def _add_image_with_border(self, img_arr_28x28: np.ndarray):
        """将 28x28 灰度图片静态加入场景，并添加统一样式的白色边框。
        返回 (img_mobj, border)
        """
        from PIL import Image
        rgb = np.stack([img_arr_28x28, img_arr_28x28, img_arr_28x28], axis=-1)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb).copy()
        pil_img = Image.fromarray(rgb, mode="RGB")
        with tempconfig({"disable_caching": True}):
            img_mobj = ImageMobject(pil_img)
        img_mobj.set_height(IMG_HEIGHT).to_edge(LEFT)
        self.add(img_mobj)

        border = SurroundingRectangle(
            img_mobj,
            buff=BORDER_BUFF,
            stroke_width=BORDER_STROKE_WIDTH,
            color=BORDER_COLOR,
        )
        self.add(border)
        return img_mobj, border


    def _build_network_nodes(self, bar_container):
        base_x = bar_container.get_right()[0] + RIGHT_GAP
        base_y = bar_container.get_center()[1]
        neurons = VGroup()
        for j in range(self.hidden_size):
            cy = base_y + ((7/2 - j)*NEURON_SPACING)
            circle = Circle(radius=NEURON_R).set_stroke(WHITE, 1.5).set_fill(YELLOW, opacity=0)
            circle.move_to([base_x, cy, 0]); neurons.add(circle)
        self.play(FadeIn(neurons), run_time=0.3)

        out_x = base_x + LAYER_GAP; out_y = base_y
        out_neurons = VGroup()
        for k in range(self.output_size):
            cy = out_y + ((9/2 - k)*OUT_SPACING)
            oc = Circle(radius=OUT_NEURON_R).set_stroke(WHITE, 1.5).set_fill(YELLOW, opacity=0)
            oc.move_to([out_x, cy, 0]); out_neurons.add(oc)
        self.play(FadeIn(out_neurons), run_time=0.3)
        return neurons, out_neurons

    def show_epoch_title(self, epoch: int, total: int | None):
        text = f"Epoch {epoch}" if total is None else f"Epoch {epoch}/{total}"
        txt = Text(text, font_size=30, color=self.colors[-1])\
              .to_corner(UL).shift(DOWN*0.2 + RIGHT*0.2)
        if not hasattr(self, "_epoch_tex"):
            self._epoch_tex = txt
            self.play(FadeIn(self._epoch_tex), run_time=0.2)
        else:
            self.play(Transform(self._epoch_tex, txt), run_time=0.2)

    # ----- 连线 -----
    def build_lines_W1(self, W1: np.ndarray, animated=True):
        lines = VGroup()
        idx_in_per_hidden = []
        W1_abs = np.abs(W1)
        W1_abs_max = np.maximum(W1_abs.max(axis=0), 1e-9)

        for j in range(self.hidden_size):
            idxs = np.argsort(-W1_abs[:, j])[:self.topk_w1]
            idx_in_per_hidden.append(idxs)
            dst = self.neurons[j].get_left()
            for i in idxs:
                src = self.targets[i].get_right()
                ctrl1 = np.array([src[0] + CURVE_BEND, src[1], 0])
                midx  = (src[0]*2 + dst[0]) / 3
                ctrl2 = np.array([midx, dst[1], 0])

                w = float(W1[i, j])
                mag01 = float(W1_abs[i, j] / W1_abs_max[j])
                color = RED if w >= 0 else BLUE
                width = 0.5 + WEIGHT_WIDTH_MAX_1 * mag01
                opacity = 0.2 + WEIGHT_OPAC_MAX_1 * mag01

                curve = CubicBezier(src, ctrl1, ctrl2, dst).set_stroke(color, width, opacity)
                lines.add(curve)

        if animated:
            self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.01), run_time=0.4)
        else:
            self.add(lines)
        return lines, idx_in_per_hidden

    def build_lines_W2(self, W2: np.ndarray, animated=True):
        lines = VGroup()
        W2_abs = np.abs(W2)
        W2_abs_max = np.maximum(W2_abs.max(), 1e-9)
        for k in range(self.output_size):
            dst = self.out_neurons[k].get_left()
            for j in range(self.hidden_size):
                src = self.neurons[j].get_right()
                w = float(W2[j, k])
                mag01 = float(W2_abs[j, k] / W2_abs_max)
                color = RED if w >= 0 else BLUE
                width = 0.5 + WEIGHT_WIDTH_MAX_2 * mag01
                opacity = 0.2 + WEIGHT_OPAC_MAX_2 * mag01
                line = Line(src, dst).set_stroke(color, width, opacity)
                lines.add(line)
        if animated:
            self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.01), run_time=0.35)
        else:
            self.add(lines)
        return lines

    def line_w2(self, j, k):
        idx = k*self.hidden_size + j
        return self.weight_2[idx]

    # ----- 风格切换 -----
    def lines_to_weight_style(self, W1, W2):
        # W1
        W1_abs = np.abs(W1); W1_abs_max = np.maximum(W1_abs.max(axis=0), 1e-9)
        anims = []; idx = 0
        for j in range(self.hidden_size):
            idxs = self.idx_in_per_hidden[j]
            for i in idxs:
                w = float(W1[i, j])
                mag01 = float(W1_abs[i, j] / W1_abs_max[j])
                color = RED if w >= 0 else BLUE
                width = 0.5 + WEIGHT_WIDTH_MAX_1 * mag01
                opacity = 0.2 + WEIGHT_OPAC_MAX_1 * mag01
                anims.append(self.weight_1[idx].animate.set_stroke(color=color, width=width, opacity=opacity))
                idx += 1
        # W2
        W2_abs = np.abs(W2); W2_abs_max = np.maximum(W2_abs.max(), 1e-9)
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                w = float(W2[j, k])
                mag01 = float(W2_abs[j, k] / W2_abs_max)
                color = RED if w >= 0 else BLUE
                width = 0.5 + WEIGHT_WIDTH_MAX_2 * mag01
                opacity = 0.2 + WEIGHT_OPAC_MAX_2 * mag01
                anims.append(self.line_w2(j,k).animate.set_stroke(color=color, width=width, opacity=opacity))
        if anims:
            self.play(*anims, run_time=1.0)

    def contrib_in_to_hidden(self, x, W1, run_time=0.6):
        contrib_1 = x[:, None] * W1
        cabs1 = np.abs(contrib_1)
        cmax1 = np.maximum(cabs1.max(axis=0), 1e-9)
        anims = []; idx = 0
        for j in range(self.hidden_size):
            idxs = self.idx_in_per_hidden[j]
            for i in idxs:
                val = float(contrib_1[i, j])
                mag01 = float(cabs1[i, j] / cmax1[j])
                color = RED if val >= 0 else BLUE
                width = 0.5 + CONTRIB_WIDTH_MAX_1 * mag01
                opacity = 0.2 + CONTRIB_OPAC_MAX_1 * mag01
                anims.append(self.weight_1[idx].animate.set_stroke(color=color, width=width, opacity=opacity))
                idx += 1
        if anims:
            self.play(*anims, run_time=run_time)

    def contrib_hidden_to_out(self, A1, W2, run_time=0.6):
        contrib_2 = A1[:, None] * W2
        cabs2 = np.abs(contrib_2)
        cmax2 = np.maximum(cabs2.max(), 1e-9)
        anims = []
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                val = float(contrib_2[j, k])
                mag01 = float(cabs2[j, k] / cmax2)
                color = RED if val >= 0 else BLUE
                width = 0.5 + CONTRIB_WIDTH_MAX_2 * mag01
                opacity = 0.2 + CONTRIB_OPAC_MAX_2 * mag01
                anims.append(self.line_w2(j,k).animate.set_stroke(color=color, width=width, opacity=opacity))
        if anims:
            self.play(*anims, run_time=run_time)

    # ----- 前向（用于展示贡献 + 点亮） -----
    def forward_and_visualize(self, x, W1, b1, W2, b2):
        # 隐层
        Z1 = x @ W1 + b1[0]
        A1 = sigmoid(Z1)
        self.contrib_in_to_hidden(x, W1, run_time=0.5)
        self.play(*[self.neurons[j].animate.set_fill(YELLOW, opacity=float(A1[j]))
                    for j in range(self.hidden_size)], run_time=0.35)

        # 输出层 logits -> softmax 概率
        Z2 = A1 @ W2 + b2[0]
        self.contrib_hidden_to_out(A1, W2, run_time=0.7)
        P = softmax(Z2)

        out_values = VGroup()
        for k in range(self.output_size):
            val = DecimalNumber(float(P[k]), num_decimal_places=2, include_sign=False, font_size=28)
            val.next_to(self.out_neurons[k], RIGHT, buff=0.16)
            out_values.add(val)

        self.play(FadeIn(out_values), run_time=0.3)

        self.play(*[self.out_neurons[k].animate.set_fill(YELLOW, opacity=(0.15 + 0.85*float(P[k])))
                    for k in range(self.output_size)], run_time=0.5)

        return out_values, Z2


    # ----- Loss 可视化 -----
    def show_loss_list(self, out_values, losses):
        loss_x = out_values[0].get_right()[0] + LOSS_BUFF
        first_loss_y = ((out_values[4].get_center()[1] + out_values[5].get_center()[1]) / 2
                        + 7 * LOSS_SPACING)

        loss_texts = VGroup()
        for i in range(LOSS_MAX_DISPLAY):
            loss_val = float(losses[i]) if i < len(losses) else 0.0
            text = DecimalNumber(loss_val, num_decimal_places=4,
                                 font_size=LOSS_FONT_SIZE).set_color(self.colors[i])
            text.move_to([loss_x, first_loss_y - i * LOSS_SPACING, 0])
            loss_texts.add(text)
        ell = Text("...", font_size=LOSS_FONT_SIZE, color=self.colors[-1])
        ell.move_to([loss_x, first_loss_y - 14 * LOSS_SPACING, 0])
        loss_texts.add(ell)

        self.play(AnimationGroup(*[FadeIn(t) for t in loss_texts], lag_ratio=LAG_RATIO_LOSS), run_time=1.0)
        self.play(FadeOut(out_values), run_time=0.5)
        self.remove(out_values)
        return loss_texts

    def show_avg_loss(self, loss_texts, losses):
        mid_idx = len(loss_texts) // 2
        loss_center = loss_texts[mid_idx].get_center()
        pairs = []; n = len(loss_texts) - 1
        for i in range(mid_idx):
            pairs += [FadeOut(loss_texts[i]), FadeOut(loss_texts[n - i])]
        self.play(*pairs, run_time=0.35)
        self.play(FadeOut(loss_texts[mid_idx]), run_time=0.2)

        loss_circle = Circle(radius=LOSS_R).set_stroke(WHITE, 1.5).move_to(loss_center)

        lastloss = float(losses[-1]) if len(losses) else 0.0
        loss_val_tex = DecimalNumber(lastloss, num_decimal_places=4, font_size=AVELOSS_FONT_SIZE)\
                        .move_to(loss_circle.get_center())

        label = Text("Loss", font_size=20, color=self.colors[-1])\
                .next_to(loss_circle, DOWN, buff=0.12)

        self.play(Create(loss_circle), Create(loss_val_tex), FadeIn(label), run_time=0.45)

        self.remove(loss_texts)
        self.reg_epoch_objs(label)

        return loss_circle, loss_val_tex


    # ----- 反向可视化 -----
    def backward_loss_to_output(self, loss_circle, avedZ2):
        dZ2_abs = np.abs(avedZ2); dZ2_abs_max = np.maximum(dZ2_abs.max(), 1e-9)
        conn = VGroup()
        src = loss_circle.get_left()
        for k in range(self.output_size):
            dst = self.out_neurons[k].get_right()
            mag01 = float(dZ2_abs[0, k] / dZ2_abs_max)
            color = RED if avedZ2[0, k] >= 0 else BLUE
            width = 0.5 + DZ2_WIDTH_MAX * mag01
            opacity = 0.2 + DZ2_OPAC_MAX * mag01
            line = Line(src, dst).set_stroke(color, width, opacity)
            conn.add(line)
        self.play(Create(conn), run_time=0.45)
        return conn

    def backward_output_to_hidden(self, avedW2):
        dW2_abs = np.abs(avedW2); dW2_abs_max = np.maximum(dW2_abs.max(), 1e-9)
        conn = VGroup()
        for j in range(self.hidden_size):
            dst = self.neurons[j].get_right()
            for k in range(self.output_size):
                dw = float(avedW2[j, k])
                mag01 = float(dW2_abs[j, k] / dW2_abs_max)
                color = RED if dw >= 0 else BLUE
                width = 0.5 + DW2_WIDTH_MAX * mag01
                opacity = 0.2 + DW2_OPAC_MAX * mag01
                src = self.out_neurons[k].get_left()
                conn.add(Line(src, dst).set_stroke(color, width, opacity))
        self.play(Create(conn), run_time=0.5)
        return conn

    def backward_hidden_to_input_gradonly(self, avedW1):
        dW1_abs = np.abs(avedW1); dW1_abs_max = np.maximum(dW1_abs.max(axis=0), 1e-9)
        conn = VGroup()
        for j in range(self.hidden_size):
            idxs = np.argsort(-dW1_abs[:, j])[:self.topk_w1]
            src = self.neurons[j].get_left()
            for i in idxs:
                dw = float(avedW1[i, j])
                mag01 = float(dW1_abs[i, j] / dW1_abs_max[j])
                color = RED if dw >= 0 else BLUE
                width = 0.5 + DW1_WIDTH_MAX * mag01
                opacity = 0.6 + DW1_OPAC_MAX * mag01
                dst = self.targets[i].get_right()
                ctrl1 = np.array([src[0] + CURVE_BEND, src[1], 0])
                midx  = (src[0]*2 + dst[0]) / 3
                ctrl2 = np.array([midx, dst[1], 0])
                conn.add(CubicBezier(src, ctrl1, ctrl2, dst).set_stroke(color, width, opacity))
        self.play(Create(conn), run_time=0.65)
        return conn

    def show_lr_and_decay(self, *conns: VGroup):
        net_group = VGroup(self.neurons, self.out_neurons)
        lr_tex = Text(f"α = {self.learning_rate:.3f}",
                      font_size=LEARNING_RATE_FONT_SIZE,
                      color=self.colors[-1]).next_to(net_group, DOWN, buff=0.5)
        self.play(FadeIn(lr_tex), run_time=0.2)
        anims = []
        for g in conns:
            for e in g:
                w = e.get_stroke_width(); o = e.get_stroke_opacity()
                anims.append(e.animate.set_stroke(width=w/2, opacity=o/2))
        self.play(*anims, run_time=0.7)
        self.play(FadeOut(lr_tex), run_time=0.2)

    # ----- 网络压缩/展开/切换 -----
    def compact_network(self):
        self.play(
            AnimationGroup(
                self.weight_1.animate.shift(COMPACT_SHIFT).scale(COMPACT_SCALE),
                self.weight_2.animate.shift(COMPACT_SHIFT).scale(COMPACT_SCALE),
                *[self.neurons[j].animate.set_fill(YELLOW, opacity=0.0) for j in range(self.hidden_size)],
                *[self.out_neurons[k].animate.set_fill(YELLOW, opacity=0.0) for k in range(self.output_size)],
                lag_ratio=0
            ),
            run_time=0.6
        )

    def expand_network(self, keep_residuals=True):
        if keep_residuals and hasattr(self, "_epoch_bin") and len(self._epoch_bin) > 0:
            try: self.bring_to_front(self._epoch_bin)
            except: pass
        inv = 1.0/float(COMPACT_SCALE)
        self.play(
            AnimationGroup(
                self.weight_1.animate.shift(-COMPACT_SHIFT).scale(inv),
                self.weight_2.animate.shift(-COMPACT_SHIFT).scale(inv),
                lag_ratio=0
            ),
            run_time=0.45
        )

    def swap_to_new_weights(self, resW1: np.ndarray, resW2: np.ndarray):
        # 展开回原位
        self.expand_network(keep_residuals=True)

        # A) W1 需要重建（Top-K 可能变化）
        new_w1, new_idx = self.build_lines_W1(resW1, animated=False)
        tgt_opac_w1 = [l.get_stroke_opacity() for l in new_w1]
        for l in new_w1: l.set_stroke(opacity=0.0)

        # B) W2 不重建：更新 stroke
        W2_abs = np.abs(resW2); W2_abs_max = max(float(W2_abs.max()), 1e-9)
        w2_anims = []
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                w = float(resW2[j, k])
                mag01 = float(W2_abs[j, k] / W2_abs_max)
                color = RED if w >= 0 else BLUE
                width = 0.5 + WEIGHT_WIDTH_MAX_2 * mag01
                opacity = 0.2 + WEIGHT_OPAC_MAX_2 * mag01
                w2_anims.append(self.line_w2(j, k).animate.set_stroke(color=color, width=width, opacity=opacity))

        self.play(
            FadeOut(self.weight_1),
            *[new_w1[i].animate.set_stroke(opacity=tgt_opac_w1[i]) for i in range(len(tgt_opac_w1))],
            *w2_anims,
            run_time=0.6
        )
        try: self.remove(self.weight_1)
        except: pass
        self.weight_1 = new_w1
        self.idx_in_per_hidden = new_idx

    # ----- 临时对象收纳 -----
    def _init_epoch_bin(self):
        self._epoch_bin = VGroup()

    def reg_epoch_objs(self, *mobs):
        for m in mobs:
            if m is None: continue
            if isinstance(m, Mobject): 
                self._epoch_bin.add(m)
            elif isinstance(m, (list, tuple, VGroup)):
                for x in (m if isinstance(m, (list, tuple)) else m.submobjects):
                    if isinstance(x, Mobject): self._epoch_bin.add(x)

    def clear_epoch_bin(self):
        if hasattr(self, "_epoch_bin") and len(self._epoch_bin) > 0:
            self.play(FadeOut(self._epoch_bin), run_time=0.3)
            try: self.remove(self._epoch_bin)
            except: pass
        self._init_epoch_bin()
