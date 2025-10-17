# simple_mlp.py
import numpy as np
import struct

# ====== 数据加载 ======
def load_images(file):
    with open(file, 'rb') as f:
        magic, size = struct.unpack('>ii', f.read(8))
        rows, cols = struct.unpack('>ii', f.read(8))
        images = np.fromfile(f, dtype=np.uint8).reshape((size, rows * cols))
    return images / 255.0

def load_labels(file):
    with open(file, 'rb') as f:
        magic, size = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# ====== MLP ======
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # TODO: complete forward process

    def compute_loss(self, y):
        m = y.shape[0]
        eps = 1e-12
        log_probs = -np.log(self.A2[np.arange(m), y] + eps)
        return float(np.sum(log_probs) / m)

    def backward(self, X, y):
        m = X.shape[0]
        y_onehot = np.zeros_like(self.A2)
        y_onehot[np.arange(m), y] = 1.0

        dZ2 = self.A2 - y_onehot
        self.dW2 = self.A1.T @ dZ2 / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        self.dW1 = X.T @ dZ1 / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    
    def step(self):
        # TODO: complete gradient descent process
        self.W1 -= self.learning_rate * 1 # TODO: change 1 into something
        self.b1 -= self.learning_rate * 1 # TODO: change 1 into something
        self.W2 -= self.learning_rate * 1 # TODO: change 1 into something
        self.b2 -= self.learning_rate * 1 # TODO: change 1 into something

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                Xb = X[i:i + batch_size]
                yb = y[i:i + batch_size]

                self.forward(Xb)
                loss = self.compute_loss(yb)
                self.backward(Xb, yb)
                self.step()

            print(f"[Epoch {epoch+1}/{epochs}] Loss={loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save_model(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to {path}")

    def load_model(self, path):
        try:
            data = np.load(path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"{path} not found, skip loading.")

# ====== 超参数 & 主程序 ======
if __name__ == "__main__":
    TRAIN_IMG = "assets/MNIST/raw/train-images-idx3-ubyte"
    TRAIN_LBL = "assets/MNIST/raw/train-labels-idx1-ubyte"
    TEST_IMG  = "assets/MNIST/raw/t10k-images-idx3-ubyte"
    TEST_LBL  = "assets/MNIST/raw/t10k-labels-idx1-ubyte"

    X_train = load_images(TRAIN_IMG)
    y_train = load_labels(TRAIN_LBL)
    X_test = load_images(TEST_IMG)
    y_test = load_labels(TEST_LBL)

    mlp = MLP(input_size=784, hidden_size=8, output_size=10, learning_rate=0.1)

    mlp.train(X_train, y_train, epochs=8, batch_size=64)
    print("Train acc:", mlp.evaluate(X_train, y_train))
    print("Test acc :", mlp.evaluate(X_test, y_test))

    mlp.save_model("mlp_model.npz")

