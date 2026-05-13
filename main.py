# Example
from src.models.datareuploading import DataReuploadingQNN
from src.data import mnist_pca_digits



X_train, y_train, X_test, y_test = mnist_pca_digits(n_components=2, seed=42)

qnn = DataReuploadingQNN(n_qubits=2, n_layers=3, seed=42)
qnn.coherent_train(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    simulator_qubits=5,
    epochs=10, 
    time_steps=10,
    total_time=0.001,
    verbose=True
)