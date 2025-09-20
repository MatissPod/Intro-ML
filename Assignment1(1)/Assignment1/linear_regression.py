import numpy as np




class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs = 5):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # ensure 2D
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # gradient descent
        for _ in range(self.epochs):
            # predictions (linear model)
            y_pred = np.dot(X, self.weights) + self.bias

            # compute gradients (MSE derivative)
            grad_w = -(2/n_samples) * np.dot(X.T, (y - y_pred))
            grad_b = -(2/n_samples) * np.sum(y - y_pred)

            # update parameters
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            
            # record MSE loss
            mse = np.mean((y - y_pred) ** 2)
            self.losses.append(mse)
            
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (np.dot(X, self.weights).flatten() + self.bias)