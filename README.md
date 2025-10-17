# 🧠 AIA 社团第一次教学例会示例代码

这是 **南京大学 AIA 社团第一次教学例会** 的示例代码仓库。
内容涵盖一个最简版的 **MNIST 全连接神经网络（MLP）训练与可视化项目**，用于讲解神经网络基本概念、训练流程以及可视化方法。

---

## 📦 环境准备

确保你拥有一个包含 `numpy` 的 Python 环境（建议 Python ≥3.9）。

若要使用可视化功能，请额外安装 manim

---

## 🚀 基本用法（无可视化）

### 1. 下载数据集

```bash
python download.py
```

会自动下载 MNIST 数据集至 `assets/MNIST/raw/` 目录下。

### 2. 训练模型

```bash
python mlp.py
```

该脚本会在终端输出每个 epoch 的损失值，并在训练结束后保存一个简单的 MLP 模型。

---

## 🎥 可视化相关流程

### 1. 生成训练日志

```bash
python train_with_logs.py
```

该脚本在 `runs/mlp1/` 目录下记录模型训练的参数、梯度、损失、样本图像等信息。

### 2. 并行渲染训练动画

```bash
python parallel_render.py
```

会使用 Manim 渲染每个 epoch 的训练过程，输出多段视频片段。

### 3. 拼接完整视频

```bash
python concat_videos.py --src renders/mlp1
```

自动将所有 epoch 片段拼接成完整的可视化训练视频。

---

## 🔍 推理可视化（infer.py）

`infer.py` 用于展示最终模型的前向传播过程与分类结果。
可以加载训练或测试样本，也可直接使用外部灰度图片。

### 用法说明

```bash
python viz_infer.py --log-dir runs/mlp1 [选项]
```

支持三种输入模式，按优先级依次生效：

| 优先级 | 参数组合                  | 说明                                                          |
| :-: | :-------------------- | :---------------------------------------------------------- |
|  ①  | `--image-file <path>` | 任意灰度图文件（自动转为 28×28 并归一化）                                    |
|  ②  | `--index <n>`         | 从测试集 `t10k-images-idx3-ubyte` 加载指定索引                        |
|  ③  | *(默认)*                | 使用训练阶段保存的 `sample_for_viz.npy`（对应 meta 中 `viz_image_index`） |

### 示例

#### 使用外部图片

```bash
python viz_infer.py --log-dir runs/mlp1 --image-file ./my_digit.png
# 若图片是白底黑字，可以反相
python viz_infer.py --log-dir runs/mlp1 --image-file ./my_digit_white_bg.png --invert
```

#### 使用测试集图片

```bash
python viz_infer.py --log-dir runs/mlp1 --index 123
# 若默认测试集路径不存在，可手动指定：
python viz_infer.py --log-dir runs/mlp1 --index 123 --test-ubyte assets/MNIST/raw/t10k-images-idx3-ubyte
```

#### 默认模式（使用训练日志中的样本）

```bash
python viz_infer.py --log-dir runs/mlp1
```

### 可选参数

| 参数          | 说明                                                                        |
| ----------- | ------------------------------------------------------------------------- |
| `--quality` | 渲染质量（`low_quality` / `medium_quality` / `high_quality` / `fourk_quality`） |
| `--out`     | 输出视频文件名（不含扩展名）                                                            |
| `--outdir`  | 输出目录                                                                      |
| `--tmpdir`  | 渲染中间文件目录（用于并行任务）                                                          |

---


## 🏫 致谢

> 本项目用于南京大学 AIA 社团 2025 年秋季学期第一次教学例会。

> 感谢所有参与教学与贡献代码的成员！
