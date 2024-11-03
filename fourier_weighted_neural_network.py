import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    load_digits,
    fetch_california_housing
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    matthews_corrcoef,
    r2_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Ensure pandas is imported for DataFrame operations

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Loss functions and their derivatives
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12  # To prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.size

class SpectralLayer:
    def __init__(self, input_dim, output_dim, R, activation='relu', total_neurons=100):
        """
        Initializes the Spectral Layer.

        Parameters:
        - input_dim: Number of input neurons.
        - output_dim: Number of output neurons.
        - R: Range for Fourier coefficients (determines number of coefficients).
        - activation: Activation function ('relu', 'sigmoid', or 'linear').
        - total_neurons: Total number of neurons in the entire network (N).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.R = R  # Fourier coefficients from -R to R
        self.num_coeffs = 2 * R + 1  # Total number of coefficients
        self.N = total_neurons  # Total number of neurons in the network

        # Initialize Fourier coefficients randomly
        self.c = np.random.randn(self.num_coeffs) * 0.01  # Shape: (num_coeffs,)

        # Initialize biases
        self.b = np.zeros((output_dim,))

        # Activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'linear':
            self.activation = linear
            self.activation_derivative = linear_derivative
        else:
            raise ValueError("Unsupported activation function")

        # Placeholders for forward and backward pass
        self.W = None
        self.z = None
        self.a = None
        self.input = None
        self.grad_c = np.zeros_like(self.c)
        self.grad_b = np.zeros_like(self.b)

    def construct_weight_matrix(self):
        """
        Constructs the symmetric weight matrix W based on Fourier coefficients.
        """
        r = np.arange(self.output_dim).reshape(-1, 1)  # Shape: (output_dim, 1)
        s = np.arange(self.input_dim).reshape(1, -1)   # Shape: (1, input_dim)
        rs_sum = r + s  # Shape: (output_dim, input_dim)

        # Compute cosine terms for all j in [-R, R]
        j_values = np.arange(-self.R, self.R + 1)  # Shape: (num_coeffs,)
        angle = (rs_sum / (self.N - 1)) * np.pi  # Shape: (output_dim, input_dim)
        cos_terms = np.cos(np.outer(j_values, angle.flatten())).reshape(self.num_coeffs, self.output_dim, self.input_dim)

        # Sum over all j to get W
        W = np.sum(self.c[:, np.newaxis, np.newaxis] * cos_terms, axis=0)  # Shape: (output_dim, input_dim)
        return W

    def forward(self, x):
        """
        Forward pass through the spectral layer.

        Parameters:
        - x: Input data of shape (batch_size, input_dim)

        Returns:
        - Activated output of shape (batch_size, output_dim)
        """
        self.input = x  # Store for backward pass
        self.W = self.construct_weight_matrix()  # Shape: (output_dim, input_dim)
        self.z = np.dot(x, self.W.T) + self.b  # Shape: (batch_size, output_dim)
        self.a = self.activation(self.z)  # Shape: (batch_size, output_dim)
        return self.a

    def backward(self, delta):
        """
        Backward pass through the spectral layer.

        Parameters:
        - delta: Gradient of loss with respect to activated output (batch_size, output_dim)

        Returns:
        - Gradient with respect to input x (batch_size, input_dim)
        """
        batch_size = self.input.shape[0]

        # Compute derivative of activation
        da_dz = self.activation_derivative(self.z)  # Shape: (batch_size, output_dim)
        delta_z = delta * da_dz  # Shape: (batch_size, output_dim)

        # Gradients w.r.t biases
        self.grad_b = np.sum(delta_z, axis=0) / batch_size  # Shape: (output_dim,)

        # Gradients w.r.t W
        grad_W = np.dot(delta_z.T, self.input) / batch_size  # Shape: (output_dim, input_dim)

        # Gradients w.r.t Fourier coefficients c_j
        r = np.arange(self.output_dim).reshape(-1, 1)  # Shape: (output_dim, 1)
        s = np.arange(self.input_dim).reshape(1, -1)   # Shape: (1, input_dim)
        rs_sum = r + s  # Shape: (output_dim, input_dim)

        angle = (rs_sum / (self.N - 1)) * np.pi  # Shape: (output_dim, input_dim)
        j_values = np.arange(-self.R, self.R + 1)  # Shape: (num_coeffs,)
        cos_j_angle = np.cos(np.outer(j_values, angle.flatten())).reshape(self.num_coeffs, self.output_dim, self.input_dim)

        # Compute gradient w.r.t c_j
        self.grad_c += np.sum(grad_W * cos_j_angle, axis=(1, 2)) / batch_size

        # Gradient w.r.t input x to pass to previous layer
        grad_x = np.dot(delta_z, self.W)  # Shape: (batch_size, input_dim)
        return grad_x

    def update_parameters(self, learning_rate):
        """
        Updates Fourier coefficients and biases using gradient descent.

        Parameters:
        - learning_rate: Learning rate for gradient descent.
        """
        self.c -= learning_rate * self.grad_c
        self.b -= learning_rate * self.grad_b

        # Reset gradients after update
        self.grad_c = np.zeros_like(self.c)
        self.grad_b = np.zeros_like(self.b)

class SpectralNeuralNetwork:
    def __init__(self, layer_sizes, R, activations):
        """
        Initializes the Spectral Neural Network.

        Parameters:
        - layer_sizes: List containing the number of neurons in each layer (including input and output layers).
        - R: List containing the range for Fourier coefficients for each layer (must have length len(layer_sizes)-1).
        - activations: List of activation functions for each layer (excluding input layer).
        """
        assert len(layer_sizes) - 1 == len(R) == len(activations), \
            "Length of R and activations must be equal to number of layers minus one."
        
        # Compute total number of neurons in the network
        self.N = sum(layer_sizes)
        print(f"Total number of neurons in the network (N): {self.N}")

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = SpectralLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                R=R[i],
                activation=activations[i],
                total_neurons=self.N  # Pass N to each layer
            )
            self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input data of shape (batch_size, input_dim)

        Returns:
        - Output of the network
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, delta):
        """
        Backward pass through the network.

        Parameters:
        - delta: Gradient of loss with respect to the network's output
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update_parameters(self, learning_rate):
        """
        Updates all layers' parameters.

        Parameters:
        - learning_rate: Learning rate for gradient descent.
        """
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def train(self, X, y, epochs, learning_rate, loss_function='mse'):
        """
        Trains the network using gradient descent.

        Parameters:
        - X: Training data of shape (num_samples, input_dim)
        - y: Training labels of shape (num_samples, output_dim)
        - epochs: Number of training epochs
        - learning_rate: Learning rate for gradient descent
        - loss_function: 'mse' or 'bce'
        """
        for epoch in range(1, epochs + 1):
            # Forward pass
            output = self.forward(X)  # Shape: (num_samples, output_dim)

            # Compute loss and its derivative
            if loss_function == 'mse':
                loss = mse_loss(y, output)
                loss_grad = mse_loss_derivative(y, output)  # Shape: (num_samples, output_dim)
            elif loss_function == 'bce':
                loss = binary_cross_entropy(y, output)
                loss_grad = binary_cross_entropy_derivative(y, output)  # Shape: (num_samples, output_dim)
            else:
                raise ValueError("Unsupported loss function")

            # Backward pass
            self.backward(loss_grad)

            # Update parameters
            self.update_parameters(learning_rate)

            # Print loss every 100 epochs
            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Makes predictions with the network.

        Parameters:
        - X: Input data of shape (num_samples, input_dim)

        Returns:
        - Predictions of shape (num_samples, output_dim)
        """
        return self.forward(X)

# Common functions for various tasks

def load_and_preprocess(dataset_loader, binary_classification=False, test_size=0.2, random_state=1234):
    """
    Loads and preprocesses a dataset.

    Parameters:
    - dataset_loader: Function to load the dataset.
    - binary_classification: Boolean indicating if it's a binary classification task.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed used by the random number generator.

    Returns:
    - X_train, X_test, y_train, y_test: Preprocessed training and test data.
    - target_names: Names of the target classes (optional, for classification tasks).
    """
    dataset = dataset_loader()
    X = dataset.data
    y = dataset.target.reshape(-1,1)

    if binary_classification:
        # Convert to binary classification: class 0 vs. others
        y = (y == 0).astype(int).reshape(-1, 1)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optional: Class names for confusion matrix
    target_names = dataset.target_names if hasattr(dataset, 'target_names') else None

    return X_train, X_test, y_train, y_test, target_names

def train_and_evaluate_snn(X_train, y_train, X_test, y_test, spectral_dims, layer_dims, activations, epochs, learning_rate, loss_function):
    """
    Trains and evaluates a Spectral Neural Network.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Test data and labels.
    - spectral_dims: List of Fourier coefficient ranges (R) per layer.
    - layer_dims: List of neuron counts per layer.
    - activations: List of activation functions per layer.
    - epochs: Number of training epochs.
    - learning_rate: Learning rate for gradient descent.
    - loss_function: 'mse' or 'bce'.

    Returns:
    - network: Trained SNN.
    - predictions: Predictions on the test data.
    - metrics: Dictionary with evaluation metrics.
    """
    # Initialize the network
    network = SpectralNeuralNetwork(layer_dims, spectral_dims, activations)

    # Train the network
    network.train(X_train, y_train, epochs, learning_rate, loss_function=loss_function)

    # Make predictions on the test set
    predictions = network.predict(X_test)

    # Evaluate predictions
    if loss_function == 'bce':
        predictions_binary = (predictions > 0.5).astype(int)
        mcc = matthews_corrcoef(y_test, predictions_binary)
        accuracy = accuracy_score(y_test, predictions_binary)
        metrics = {'MCC': mcc, 'Accuracy': accuracy}
    else:
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        metrics = {'R² Score': r2, 'MSE': mse}

    return network, predictions, metrics

def train_and_evaluate_traditional_models(X_train, y_train, X_test, y_test, task_type='classification'):
    """
    Trains and evaluates traditional ML models.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Test data and labels.
    - task_type: 'classification' or 'regression'.

    Returns:
    - models: Dictionary of trained models.
    - predictions: Dictionary of model predictions.
    - metrics: Dictionary of evaluation metrics for each model.
    """
    models = {}
    predictions = {}
    metrics = {}

    if task_type == 'classification':
        # Initialize models
        models['SVM'] = SVC(probability=True, random_state=42)
        models['MLP'] = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train and predict
        for name, model in models.items():
            model.fit(X_train, y_train.ravel())
            predictions[name] = model.predict(X_test)

            # Calculate metrics
            mcc = matthews_corrcoef(y_test, predictions[name])
            accuracy = accuracy_score(y_test, predictions[name])
            metrics[name] = {'MCC': mcc, 'Accuracy': accuracy}

    elif task_type == 'regression':
        # Initialize models
        models['Linear Regression'] = LinearRegression()
        models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train and predict
        for name, model in models.items():
            model.fit(X_train, y_train.ravel())
            predictions[name] = model.predict(X_test).reshape(-1, 1)

            # Calculate metrics
            r2 = r2_score(y_test, predictions[name])
            mse = mean_squared_error(y_test, predictions[name])
            metrics[name] = {'R² Score': r2, 'MSE': mse}

    else:
        raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")

    return models, predictions, metrics

def compare_models(snn_metrics, traditional_metrics, task_type='classification'):
    """
    Compares SNN metrics with traditional ML models.

    Parameters:
    - snn_metrics: Dictionary of SNN evaluation metrics.
    - traditional_metrics: Dictionary of traditional models and their metrics.
    - task_type: 'classification' or 'regression'.

    Returns:
    - comparison_df: DataFrame with comparison results.
    """
    # Create DataFrame for traditional models
    traditional_df = pd.DataFrame(traditional_metrics).T

    # Add SNN metrics
    traditional_df.loc['Spectral NN'] = snn_metrics

    return traditional_df

# Plotting functions

def plot_loss(history, title):
    """
    Plots the loss curve.

    Parameters:
    - history: List of loss values.
    - title: Title of the plot.
    """
    plt.figure(figsize=(8,6))
    plt.plot(history, label='Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_classification_comparison(comparison_df, title):
    """
    Plots a bar chart comparing MCC scores across models.

    Parameters:
    - comparison_df: DataFrame with comparison results.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10,6))
    sns.barplot(x=comparison_df.index, y='MCC', data=comparison_df, palette='viridis')
    plt.title(title)
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.ylim(-1,1)
    plt.xticks(rotation=45)
    plt.show()

def plot_regression_comparison(comparison_df, title):
    """
    Plots a bar chart comparing R² scores across models.

    Parameters:
    - comparison_df: DataFrame with comparison results.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10,6))
    sns.barplot(x=comparison_df.index, y='R² Score', data=comparison_df, palette='magma')
    plt.title(title)
    plt.ylabel('R² Score')
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.show()

def plot_confusion_matrix(cm, classes, title):
    """
    Plots the confusion matrix.

    Parameters:
    - cm: Confusion matrix.
    - classes: List of class names.
    - title: Title of the plot.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_scores, title):
    """
    Plots the ROC curve.

    Parameters:
    - y_true: True binary labels.
    - y_scores: Scores/probabilities from the classifier.
    - title: Title of the plot.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_regression_predictions(y_true, y_pred, title):
    """
    Plots predicted vs actual values for regression.

    Parameters:
    - y_true: True target values.
    - y_pred: Predicted target values.
    - title: Title of the plot.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.7, label='Spectral NN')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Individual functions for each dataset

def process_breast_cancer():
    print("\n=== Breast Cancer Classification ===")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess(
        load_breast_cancer, binary_classification=True
    )

    # Spectral dimensions (R) and layer dimensions (number of neurons)
    R = [31, 31]  # [R_hidden, R_output]
    layer_dims = [X_train.shape[1], 16, 1]  # [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'sigmoid']
    print(f"Spectral Ranges: {R}, Layer Dimensions: {layer_dims}, Activations: {activations}")

    # Train and evaluate the SNN
    print("Training Spectral Neural Network...")
    snn, snn_predictions, snn_metrics = train_and_evaluate_snn(
        X_train, y_train, X_test, y_test,
        spectral_dims=R,  # Pass R correctly
        layer_dims=layer_dims,
        activations=activations,
        epochs=15000,
        learning_rate=50,
        loss_function='bce'
    )

    # Train and evaluate traditional ML models
    print("Training Traditional ML Models...")
    _, traditional_predictions, traditional_metrics = train_and_evaluate_traditional_models(
        X_train, y_train, X_test, y_test, task_type='classification'
    )

    # Compare models
    comparison_df = compare_models(snn_metrics, traditional_metrics, task_type='classification')
    print("\nModel Comparison:")
    print(comparison_df)

    # Plot comparison
    plot_classification_comparison(comparison_df, 'MCC Comparison on Breast Cancer Classification')

    # Additional Visualizations
    predictions_binary = (snn_predictions > 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions_binary)
    plot_confusion_matrix(cm, classes=target_names, title='Confusion Matrix for Breast Cancer Classification')

    roc_scores = snn_predictions.ravel()
    plot_roc_curve(y_test, roc_scores, title='ROC Curve for Breast Cancer Classification')

def process_diabetes():
    print("\n=== Diabetes Regression ===")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, _ = load_and_preprocess(
        load_diabetes, binary_classification=False
    )

    # Spectral dimensions (R) and layer dimensions (number of neurons)
    R = [63, 63]  # [R_hidden, R_output]
    layer_dims = [X_train.shape[1], 16, 1]  # [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'linear']
    print(f"Spectral Ranges: {R}, Layer Dimensions: {layer_dims}, Activations: {activations}")

    # Train and evaluate the SNN
    print("Training Spectral Neural Network...")
    snn, snn_predictions, snn_metrics = train_and_evaluate_snn(
        X_train, y_train, X_test, y_test,
        spectral_dims=R,  # Pass R correctly
        layer_dims=layer_dims,
        activations=activations,
        epochs=1000,
        learning_rate=1,
        loss_function='mse'
    )

    # Train and evaluate traditional ML models
    print("Training Traditional ML Models...")
    _, traditional_predictions, traditional_metrics = train_and_evaluate_traditional_models(
        X_train, y_train, X_test, y_test, task_type='regression'
    )

    # Compare models
    comparison_df = compare_models(snn_metrics, traditional_metrics, task_type='regression')
    print("\nModel Comparison:")
    print(comparison_df)

    # Plot comparison
    plot_regression_comparison(comparison_df, 'R² Score Comparison on Diabetes Regression')

    # Additional Visualizations
    plot_regression_predictions(y_test, snn_predictions, "Predicted vs Actual Values for Diabetes Regression")

def process_iris():
    print("\n=== Iris Classification ===")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess(
        load_iris, binary_classification=True
    )

    # Spectral dimensions (R) and layer dimensions (number of neurons)
    R = [3, 1]  # [R_hidden, R_output]
    layer_dims = [X_train.shape[1], 8, 1]  # [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'sigmoid']
    print(f"Spectral Ranges: {R}, Layer Dimensions: {layer_dims}, Activations: {activations}")

    # Train and evaluate the SNN
    print("Training Spectral Neural Network...")
    snn, snn_predictions, snn_metrics = train_and_evaluate_snn(
        X_train, y_train, X_test, y_test,
        spectral_dims=R,  # Pass R correctly
        layer_dims=layer_dims,
        activations=activations,
        epochs=5000,
        learning_rate=10,
        loss_function='bce'
    )

    # Train and evaluate traditional ML models
    print("Training Traditional ML Models...")
    _, traditional_predictions, traditional_metrics = train_and_evaluate_traditional_models(
        X_train, y_train, X_test, y_test, task_type='classification'
    )

    # Compare models
    comparison_df = compare_models(snn_metrics, traditional_metrics, task_type='classification')
    print("\nModel Comparison:")
    print(comparison_df)

    # Plot comparison
    plot_classification_comparison(comparison_df, 'MCC Comparison on Iris Classification')

    # Additional Visualizations
    predictions_binary = (snn_predictions > 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions_binary)
    plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], title='Confusion Matrix for Iris Classification')

    roc_scores = snn_predictions.ravel()
    plot_roc_curve(y_test, roc_scores, title='ROC Curve for Iris Classification')

def process_wine():
    print("\n=== Wine Classification ===")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess(
        load_wine, binary_classification=True
    )

    # Spectral dimensions (R) and layer dimensions (number of neurons)
    R = [4, 1]  # [R_hidden, R_output]
    layer_dims = [X_train.shape[1], 16, 1]  # [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'sigmoid']
    print(f"Spectral Ranges: {R}, Layer Dimensions: {layer_dims}, Activations: {activations}")

    # Train and evaluate the SNN
    print("Training Spectral Neural Network...")
    snn, snn_predictions, snn_metrics = train_and_evaluate_snn(
        X_train, y_train, X_test, y_test,
        spectral_dims=R,  # Pass R correctly
        layer_dims=layer_dims,
        activations=activations,
        epochs=5000,
        learning_rate=10,
        loss_function='bce'
    )

    # Train and evaluate traditional ML models
    print("Training Traditional ML Models...")
    _, traditional_predictions, traditional_metrics = train_and_evaluate_traditional_models(
        X_train, y_train, X_test, y_test, task_type='classification'
    )

    # Compare models
    comparison_df = compare_models(snn_metrics, traditional_metrics, task_type='classification')
    print("\nModel Comparison:")
    print(comparison_df)

    # Plot comparison
    plot_classification_comparison(comparison_df, 'MCC Comparison on Wine Classification')

    # Additional Visualizations
    predictions_binary = (snn_predictions > 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions_binary)
    plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], title='Confusion Matrix for Wine Classification')

    roc_scores = snn_predictions.ravel()
    plot_roc_curve(y_test, roc_scores, title='ROC Curve for Wine Classification')

def process_digits():
    print("\n=== Digits Classification ===")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess(
        load_digits, binary_classification=True
    )

    # Spectral dimensions (R) and layer dimensions (number of neurons)
    R = [31, 31]  # [R_hidden, R_output]
    layer_dims = [X_train.shape[1], 32, 1]  # [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'sigmoid']
    print(f"Spectral Ranges: {R}, Layer Dimensions: {layer_dims}, Activations: {activations}")

    # Train and evaluate the SNN
    print("Training Spectral Neural Network...")
    snn, snn_predictions, snn_metrics = train_and_evaluate_snn(
        X_train, y_train, X_test, y_test,
        spectral_dims=R,  # Pass R correctly
        layer_dims=layer_dims,
        activations=activations,
        epochs=5000,
        learning_rate=50.0,
        loss_function='bce'
    )

    # Train and evaluate traditional ML models
    print("Training Traditional ML Models...")
    _, traditional_predictions, traditional_metrics = train_and_evaluate_traditional_models(
        X_train, y_train, X_test, y_test, task_type='classification'
    )

    # Compare models
    comparison_df = compare_models(snn_metrics, traditional_metrics, task_type='classification')
    print("\nModel Comparison:")
    print(comparison_df)

    # Plot comparison
    plot_classification_comparison(comparison_df, 'MCC Comparison on Digits Classification')

    # Additional Visualizations
    predictions_binary = (snn_predictions > 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions_binary)
    plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], title='Confusion Matrix for Digits Classification')

    roc_scores = snn_predictions.ravel()
    plot_roc_curve(y_test, roc_scores, title='ROC Curve for Digits Classification')

def process_california_housing():
    print("\n=== California Housing Regression ===")

    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test, _ = load_and_preprocess(
        fetch_california_housing, binary_classification=False
    )

    # Spectral dimensions (R) and layer dimensions (number of neurons)
    R = [32, 32]  # [R_hidden, R_output]
    layer_dims = [X_train.shape[1], 16, 1]  # [input_dim, hidden_dim, output_dim]
    activations = ['relu', 'linear']
    print(f"Spectral Ranges: {R}, Layer Dimensions: {layer_dims}, Activations: {activations}")

    # Train and evaluate the SNN
    print("Training Spectral Neural Network...")
    snn, snn_predictions, snn_metrics = train_and_evaluate_snn(
        X_train, y_train, X_test, y_test,
        spectral_dims=R,  # Pass R correctly
        layer_dims=layer_dims,
        activations=activations,
        epochs=10000,
        learning_rate=5000,
        loss_function='mse'
    )

    # Train and evaluate traditional ML models
    print("Training Traditional ML Models...")
    _, traditional_predictions, traditional_metrics = train_and_evaluate_traditional_models(
        X_train, y_train, X_test, y_test, task_type='regression'
    )

    # Compare models
    comparison_df = compare_models(snn_metrics, traditional_metrics, task_type='regression')
    print("\nModel Comparison:")
    print(comparison_df)

    # Plot comparison
    plot_regression_comparison(comparison_df, 'R² Score Comparison on California Housing Regression')

    # Additional Visualizations
    plot_regression_predictions(y_test, snn_predictions, "Predicted vs Actual Values for California Housing Regression")

# Main function to call individual dataset processing functions
def main():
    # Uncomment the datasets you wish to process
    process_breast_cancer()
    process_diabetes()
    process_iris()
    process_wine()
    process_digits()
    process_california_housing()

if __name__ == "__main__":
    main()

