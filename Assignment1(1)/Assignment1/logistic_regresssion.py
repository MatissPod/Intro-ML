import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, learning_rate=0.001, epochs=100):
        # NOTE: Feel free to add any hyperparameters
        # (with defaults) as you see fit
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights, self.bias = None, None
        self.losses, self.accuracies = [], []

    def _sigmoid_func(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        loss = np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
        return loss

    def _compute_gradients(self, x, y, y_pred):
        """
        Calculates the gradient for weights and bias for logistic regression
        Returns gradient_weights, gradient_bias
        """
        n = len(y)
        grad_w = -np.dot(x.T, (y - y_pred)) / n
        grad_b = -np.mean(y - y_pred)
        return grad_w, grad_b

    def _compute_accuracy(self, y, y_pred):
        """
        Computes accuracy for logistic regression
        returns accuracy
        """
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y == y_pred_binary)
        return accuracy

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        # print(f"weights -= {self.learning_rate} * {grad_w}")
        self.bias -= self.learning_rate * grad_b
        # print(f"bias -= {self.learning_rate} * {grad_b}")

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples)
                n columns (#features)
            y (array<m>): a vector of floats
        """

        # Initialize weights and bias
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
        if self.bias is None:
            self.bias = 0.0

        # Gradient descent
        for i in range(self.epochs):
            y_pred = self._sigmoid_func(np.dot(X, self.weights) + self.bias)

            grad_w, grad_b = self._compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss_ = self._compute_loss(y, y_pred)
            self.losses.append(loss_)
            accuracy_ = self._compute_accuracy(y, y_pred)
            self.accuracies.append(accuracy_)

    def plot_roc_curve(self, X, y):
        """
        Plots the ROC curve

        Args:
            X (array<m,n>): input features
            y (array<m>): true labels
        """
        y_pred_proba_lin = np.dot(X, self.weights) + self.bias
        y_pred_proba = self._sigmoid_func(y_pred_proba_lin)
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of integers (0 or 1)
        """
        y_pred = self._sigmoid_func(np.dot(X, self.weights) + self.bias)
        return (y_pred >= 0.5).astype(int)