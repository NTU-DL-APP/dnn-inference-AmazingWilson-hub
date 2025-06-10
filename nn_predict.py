import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # x shape: (batch_size, num_classes)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    


if __name__ == "__main__":
    from utils.mnist_reader import load_mnist

    # 讀取模型架構
    with open("model/fashion_mnist.json", "r") as f:
        model_arch = json.load(f)

    # 讀取模型權重
    weights_npz = np.load("model/fashion_mnist.npz")
    weights = {k: weights_npz[k] for k in weights_npz.files}

    # 讀取測試資料
    X_test, y_test = load_mnist("data", kind="t10k")
    X_test = X_test.astype("float32") / 255.0

    # 跑前向推論
    preds = nn_inference(model_arch, weights, X_test)
    pred_labels = np.argmax(preds, axis=1)

    # 計算準確率
    acc = np.mean(pred_labels == y_test)
    print(f"✅ NumPy 推論 Accuracy: {acc:.4f}")
