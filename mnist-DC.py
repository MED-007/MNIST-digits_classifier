import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x,y =fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
x=x/255.0

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=15, alpha=1e-4, solver='adam', verbose=10, random_state=1)
mlp.fit(x_train_scaled, y_train)
y_pred = mlp.predict(x_test_scaled)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
def plot_digits(x, y_true, y_pred=None, n=10):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        ax=plt.subplot(2, n//2, i + 1)
        ax.imshow(x[i].reshape(28, 28), cmap='gray')
        title =f"Label: {y_true[i]}"
        if y_pred is not None:
            title += "\nPred: " + str(y_pred[i])
            if y_true[i]==y_pred[i]:
                color ='green'
            else:
                color ='red'
            ax.set_title(title, color=color, fontsize=8)

        else:
            ax.set_title(title, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plot_digits(x_test, y_test, y_pred)
