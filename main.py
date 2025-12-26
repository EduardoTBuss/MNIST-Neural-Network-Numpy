from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random
import os

LR = 0.006
batch_size = 16
epochs = 35
weight_decay = 0.001
dropout1 = 0.1
dropout2 = 0.1
dropoutlayer = True

input_size = 784
hidden1 = 64
hidden2 = 32
output_size = 10

def ReLU(Z):
    return np.maximum(0, Z)

def dReLU(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(y):
    oh = np.zeros((y.size, 10))
    oh[np.arange(y.size), y] = 1
    return oh.T 

def Load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).T / 255.0 
    x_test = x_test.reshape(-1, 784).T / 255.0
    return x_train, y_train, x_test, y_test

def Create_weight():
    W1 = np.random.randn(hidden1, input_size) * np.sqrt(2 / input_size)
    W2 = np.random.randn(hidden2, hidden1) * np.sqrt(2 / hidden1)
    W3 = np.random.randn(output_size, hidden2) * np.sqrt(2 / hidden2)
    
    B1 = np.zeros((hidden1, 1))
    B2 = np.zeros((hidden2, 1))
    B3 = np.zeros((output_size, 1))
    
    return W1, B1, W2, B2, W3, B3

def forward(X, W1, B1, W2, B2, W3, B3, dropout1 = 0, dropout2 = 0, dropoutlayer = False):
    
    Z1 = W1 @ X + B1
    A1 = ReLU(Z1)
    
    if dropoutlayer:
        M1 = (np.random.rand(*A1.shape) > dropout1).astype(float)
        A1 = (A1*M1)/(1-dropout1)
    else:
        M1 = None
        
    Z2 = W2 @ A1 + B2
    A2 = ReLU(Z2)
    
    if dropoutlayer:
        M2 = (np.random.rand(*A2.shape) > dropout2).astype(float)
        A2 = (A2*M2) / (1-dropout2)
    else:
        M2 = None
        
    Z3 = W3 @ A2 + B3
    A3 = softmax(Z3)
    
    return Z1, A1, M1, Z2, A2, M2, Z3, A3

def backward(X, Y, Z1, A1, M1, Z2, A2, M2, Z3, A3, W1, W2, W3, dropout1=0.0, dropout2=0.0):
    m = X.shape[1]
    
    dZ3 = A3 - Y
    dW3 = (1/m) * dZ3 @ A2.T
    dB3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = W3.T @ dZ3
    
    if M2 is not None:
        dA2 = dA2 * M2 / (1 - dropout2)
        
    dZ2 = dA2 * dReLU(Z2)
    dW2 = (1/m) * dZ2 @ A1.T
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = W2.T @ dZ2
    
    if M1 is not None:
        dA1 = dA1 * M1 / (1 - dropout1)
        
    dZ1 = dA1 * dReLU(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    dB1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, dB1, dW2, dB2, dW3, dB3

def update(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, weight_decay, m):
    W1 -= LR * (dW1 + (weight_decay/m * W1))
    W2 -= LR * (dW2 + (weight_decay/m * W2))
    W3 -= LR * (dW3 + (weight_decay/m * W3))
    
    B1 -= LR * dB1
    B2 -= LR * dB2
    B3 -= LR * dB3
    
    return W1, B1, W2, B2, W3, B3

def compute_loss(Y, A3, W1, W2, W3, weight_decay):
    m = Y.shape[1]
    cross_entropy = -np.sum(Y * np.log(A3 + 1e-8)) / m
    
    l2 = (weight_decay/(2*m)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    
    return cross_entropy + l2



def show_image(x_test, y_test, W1, B1, W2, B2, W3, B3, show_prob=False, color=True, only_errors=False):
    if only_errors:
        indices = []

        for i in range(x_test.shape[1]):
            _, _, _, _, _, _, _, pred = forward(
                x_test[:, i].reshape(-1, 1),
                W1, B1, W2, B2, W3, B3
            )
            predicted_label = int(np.argmax(pred, axis=0)[0])
            if predicted_label != int(y_test[i]):
                indices.append(i)

        if len(indices) == 0:
            print("Nenhum erro encontrado!")
            return

        indices = random.sample(indices, min(9, len(indices)))

    else:
        
        indices = random.sample(range(x_test.shape[1]), 9)

    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(indices):
        img = x_test[:, idx].reshape(28, 28)
        true_label = int(y_test[idx])

        _, _, _, _, _, _, _, pred_single = forward(
            x_test[:, idx].reshape(-1, 1),
            W1, B1, W2, B2, W3, B3
        )

        predicted_label = int(np.argmax(pred_single, axis=0)[0])
        prob = float(pred_single[predicted_label, 0])

        title = f"Label: {true_label}\nPredict: {predicted_label}"
        if show_prob:
            title += f" ({prob*100:.1f}%)"

        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        if color:
            c = 'green' if predicted_label == true_label else 'red'
            plt.title(title, color=c)
        else:
            plt.title(title)

    plt.tight_layout()
    plt.show()



x_train, y_train, x_test, y_test = Load_dataset()
m = x_train.shape[1]
W1, B1, W2, B2, W3, B3 = Create_weight()

losses = []

for epoch in range(epochs):
    permutation = np.random.permutation(m)
    X_shuffled = x_train[:, permutation]
    Y_shuffled = one_hot(y_train[permutation])

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        Y_batch = Y_shuffled[:, i:i+batch_size]

        Z1, A1, M1, Z2, A2, M2, Z3, A3 = forward(
            X_batch, W1, B1, W2, B2, W3, B3,
            dropout1,
            dropout2,
            dropoutlayer
        )

        dW1, dB1, dW2, dB2, dW3, dB3 = backward(
            X_batch, Y_batch,
            Z1, A1, M1, Z2, A2, M2, Z3, A3,
            W1, W2, W3,
            dropout1,
            dropout2
        )

        W1, B1, W2, B2, W3, B3 = update(
            W1, B1, W2, B2, W3, B3,
            dW1, dB1, dW2, dB2, dW3, dB3,
            weight_decay, batch_size
        )

    Z1, A1, M1, Z2, A2, M2, Z3, A3 = forward(
        x_train, W1, B1, W2, B2, W3, B3
    )

    Y_train_full = one_hot(y_train)
    loss = compute_loss(Y_train_full, A3, W1, W2, W3, weight_decay)
    losses.append(loss)

    plt.figure()
    plt.plot(losses)
    plt.title(f"Epoca: {epoch+1}")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"grafico_loss.png")
    plt.close()

    predictions = np.argmax(A3, axis=0)
    acc = np.mean(predictions == y_train) * 100
    print(f"Época {epoch+1}/{epochs} | Loss: {loss:.4f} | Acurácia: {acc:.2f}%")


Z1, A1, M1, Z2, A2, M2, Z3, A3 = forward(x_test, W1, B1, W2, B2, W3, B3)
Yt = one_hot(y_test)
loss_test = compute_loss(Yt, A3 , W1, W2, W3, weight_decay)
predictions = np.argmax(A3, axis=0)
test_acc = np.mean(predictions == y_test) * 100

print(f"\nAcurácia final no conjunto de teste: {test_acc:.2f}% | Loss teste: {loss_test:.4f}")

show_image(x_test, y_test, W1, B1, W2, B2, W3, B3, show_prob=True)
show_image(x_test, y_test, W1, B1, W2, B2, W3, B3, show_prob=True, only_errors=True)


