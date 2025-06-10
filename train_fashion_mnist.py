import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# === 載入 Fashion-MNIST 資料集 ===
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# === 建立模型 ===
model = Sequential([
    Flatten(input_shape=(28, 28)),        # 將 28x28 展平為 784
    Dense(128, activation="relu"),        # 隱藏層
    Dense(10, activation="softmax")       # 輸出層，10 類別
])

# === 編譯模型 ===
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# === 訓練模型 ===
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# === 儲存為 .h5 模型檔 ===
model.save("fashion_mnist.h5")

print("✅ 訓練完成，模型已儲存為 fashion_mnist.h5")
