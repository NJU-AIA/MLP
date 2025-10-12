# trainer_with_logs.py (clean)
import os, json, time
import numpy as np

from mlp import MLP, load_images, load_labels

# ====== 全局配置 ======
TRAIN_IMAGES = "assets/MNIST/raw/train-images-idx3-ubyte"
TRAIN_LABELS = "assets/MNIST/raw/train-labels-idx1-ubyte"
INPUT_SIZE   = 784
HIDDEN_SIZE  = 8
OUTPUT_SIZE  = 10
LEARNING_RATE = 0.1
EPOCHS       = 8
BATCH_SIZE   = 64
TOPK_W1      = 50            # 仅写入 meta，viz 不强依赖
LOG_DIR      = "runs/mlp1"   # 保存日志的目录，None 则自动创建
SAVE_EVERY   = 1
VIZ_INDEX    = 4

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def now_run_dir(prefix="mlp"):
    s = time.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("runs", f"{prefix}-{s}")

# 与 simple_mlp 的 forward/backward 一致；外部算梯度以便记录
def step_and_collect(mlp: MLP, Xb: np.ndarray, yb: np.ndarray):
    # 前向
    A2 = mlp.forward(Xb)

    # 交叉熵损失（平均）
    m = yb.shape[0]
    eps = 1e-12
    loss = float(np.sum(-np.log(A2[np.arange(m), yb] + eps)) / m)

    # one-hot
    y1h = np.zeros_like(A2)
    y1h[np.arange(m), yb] = 1.0

    # 反向
    dZ2 = A2 - y1h                             # (m,10)
    dW2 = (mlp.A1.T @ dZ2) / m                 # (H,10)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ mlp.W2.T                       # (m,H)
    s = 1.0 / (1.0 + np.exp(-mlp.Z1))          # sigmoid(Z1)
    dZ1 = dA1 * (s * (1.0 - s))                # (m,H)
    dW1 = (Xb.T @ dZ1) / m                     # (784,H)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # 参数更新
    lr = mlp.learning_rate
    mlp.W1 -= lr * dW1
    mlp.b1 -= lr * db1
    mlp.W2 -= lr * dW2
    mlp.b2 -= lr * db2

    return loss, dW1, dW2, dZ2

def train_and_log(
    train_images, train_labels,
    input_size=784, hidden_size=8, output_size=10,
    learning_rate=0.1, epochs=8, batch_size=64,
    topk_w1=50,
    log_dir=None, save_every=1, viz_index=None
):
    # 数据
    X_train = load_images(train_images)
    y_train = load_labels(train_labels)

    # 日志目录
    if log_dir is None:
        log_dir = now_run_dir("mlp")
    ensure_dir(log_dir)

    # 可视化样例索引与存盘（28x28, 值域[0,1]）
    if viz_index is None:
        rng = np.random.default_rng(0)
        viz_index = int(rng.integers(low=0, high=X_train.shape[0]))
    else:
        viz_index = int(viz_index)
    np.save(os.path.join(log_dir, "sample_for_viz.npy"),
            X_train[viz_index].reshape(28, 28))

    # meta.json（保留必要字段）
    meta = dict(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
        topk_w1=topk_w1,
        viz_image_index=viz_index
    )
    with open(os.path.join(log_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 模型
    mlp = MLP(input_size, hidden_size, output_size, learning_rate)

    # 训练
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    for ep in range(1, epochs + 1):
        # 记录 epoch 开始的权重（ori*）
        oriW1, orib1 = mlp.W1.copy(), mlp.b1.copy()
        oriW2, orib2 = mlp.W2.copy(), mlp.b2.copy()

        # 用于统计 epoch 平均梯度/残差与 loss 序列
        sum_dW1 = np.zeros_like(mlp.W1)
        sum_dW2 = np.zeros_like(mlp.W2)
        sum_dZ2 = np.zeros((1, output_size))
        losses = []

        for i in range(0, n_samples, batch_size):
            Xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]

            loss, dW1, dW2, dZ2 = step_and_collect(mlp, Xb, yb)

            # 统计（对 batch 求平均后，再对 batch 做均值 -> 这里用求和，最后再 / n_batches）
            losses.append(loss)
            sum_dW1 += dW1
            sum_dW2 += dW2
            sum_dZ2 += np.sum(dZ2, axis=0, keepdims=True)  # 与旧实现保持一致语义

        # 计算 epoch 级平均量
        avedW1 = sum_dW1 / n_batches
        avedW2 = sum_dW2 / n_batches
        avedZ2 = sum_dZ2 / n_batches

        # 记录 epoch 结束的权重（res*）
        resW1, resb1 = mlp.W1.copy(), mlp.b1.copy()
        resW2, resb2 = mlp.W2.copy(), mlp.b2.copy()

        # 打印“最后一次 batch 的 loss”
        last_loss = float(losses[-1]) if losses else 0.0
        print(f"[Epoch {ep}/{epochs}] last_loss={last_loss:.4f}")

        # ✅ 写出该 epoch 的快照（仅保留可视所需字段）
        np.savez(
            os.path.join(log_dir, f"epoch_{ep:03d}.npz"),
            oriW1=oriW1, orib1=orib1, oriW2=oriW2, orib2=orib2,
            resW1=resW1, resb1=resb1, resW2=resW2, resb2=resb2,
            avedW1=avedW1, avedW2=avedW2,
            avedZ2=avedZ2,
            losses=np.array(losses, dtype=np.float32)
        )

        # 可选 checkpoint
        if save_every and (ep % save_every == 0):
            mlp.save_model(os.path.join(log_dir, f"model_epoch_{ep:03d}.npz"))

    # 最终模型
    mlp.save_model(os.path.join(log_dir, "model_last.npz"))
    print(f"Logs & checkpoints saved to: {log_dir}")

if __name__ == "__main__":
    train_and_log(
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        topk_w1=TOPK_W1,
        log_dir=LOG_DIR,
        save_every=SAVE_EVERY,
        viz_index=VIZ_INDEX,
    )
