
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))

def train_logistic_regression(X, y, num_classes, learning_rate=0.1, epochs=100):
    """
    Trains logistic regression models for multi-class classification using one-vs-all approach.

    Args:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Labels.
        num_classes (int): Number of classes.
        learning_rate (float): Learning rate.
        epochs (int): Number of training epochs.

    Returns:
        numpy.ndarray: Weights for each class.
    """
    m, n = X.shape
    weights = np.zeros((num_classes, n))

    for class_label in range(num_classes):
        y_binary = (y == class_label).astype(int)
        for epoch in range(epochs):
            predictions = sigmoid(np.dot(X, weights[class_label].T))
            gradient = np.dot((predictions - y_binary).T, X) / m
            weights[class_label] -= learning_rate * gradient

    return weights

def predict(X, weights):
    logits = np.dot(X, weights.T)
    return np.argmax(logits, axis=1)

def evaluate_model(X, y, weights):
    predictions = predict(X, weights)
    accuracy = np.mean(predictions == y)
    return accuracy

def plot_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()