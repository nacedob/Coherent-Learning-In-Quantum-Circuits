import numpy as np
import pytest
from src.data import mnist_pca_digits, corners3d
from src.models.datareuploading import DataReuploadingQNN

def test_qnn_training():
    # ---------------------------
    # Generate dataset
    # ---------------------------
    X_train, y_train, X_test, y_test = mnist_pca_digits(
        n_components=3,
        n_train=100,
        n_test=50,
        seed=123,
    )

    # ---------------------------
    # Initialize QNN
    # ---------------------------
    n_qubits = 1
    n_layers = 2
    qnn = DataReuploadingQNN(n_qubits=n_qubits, n_layers=n_layers, seed=42)

    # ---------------------------
    # Initial loss and accuracy
    # ---------------------------
    initial_loss = qnn.loss(X_train, y_train)
    initial_acc = qnn.accuracy(X_train, y_train)

    print(f"Initial loss={initial_loss:.4f}, acc={initial_acc:.4f}")

    # ---------------------------
    # Train the QNN
    # ---------------------------
    qnn.train(
        X_train,
        y_train,
        epochs=50,
        learning_rate=0.1,
        X_test=X_test,
        y_test=y_test,
        print_every=10
    )

    # ---------------------------
    # Post-training loss and accuracy
    # ---------------------------
    final_loss = qnn.loss(X_train, y_train)
    final_acc = qnn.accuracy(X_train, y_train)

    print(f"Final loss={final_loss:.4f}, acc={final_acc:.4f}")

    # ---------------------------
    # Simple assertions
    # ---------------------------
    assert final_loss < initial_loss, "Training did not reduce loss"
    assert final_acc > initial_acc, "Training did not improve accuracy"

    # Optional: Check test accuracy is reasonable
    test_acc = qnn.accuracy(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    assert 0.0 <= test_acc <= 1.0
    
    
def test_qnn_and_corners_as_expected():
    qnn = DataReuploadingQNN(n_qubits=2, n_layers=2)
    X, y = corners3d(n_points=1000)
    
    qnn.train(
        X,
        y,
        epochs=500,
        learning_rate=0.1,
        print_every=10
    )
    
    acc = qnn.accuracy(X, y)
    
    assert acc > 0.8
    
if __name__ == "__main__":
    test_qnn_and_corners_as_expected()