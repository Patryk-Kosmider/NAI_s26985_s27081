import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, fashion_mnist

DEFAULT_EPOCHS = 10


def create_mlp_model(input_dim, output_classes, hidden_layers=[64, 32]):
    """
    Przygotowanie modelu w pełni połączonej sieci neuronowej (MLP).
    :param input_dim: Rozmiar wejścia
    :param output_classes: Klasy wyjściowe
    :param hidden_layers: Warstwa ukryte
    :return: Model Keras
    """
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(output_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_cnn_model(input_shape, output_classes, filters=[32, 64]):
    """
    Przygotowanie modelu konwolucyjnej sieci neuronowej (CNN).
    :param input_shape: Rozmiar wejścia
    :param output_classes: Klasy wyjściowe
    :param filters: Ilość filtrów w warstwach konwolucyjnych
    :return: Model Keras
    """
    model = Sequential()

    model.add(
        Conv2D(
            filters[0],
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D((2, 2)))

    for num_filters in filters[1:]:
        model.add(Conv2D(num_filters, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def plot_confusion_matrix(model, X_test, y_test_true, title, class_names=None):
    """
    Wyświetla macierz pomyłek oraz raport klasyfikacji dla podanego modelu i danych testowych.
    :param model: Model Keras
    :param X_test: Zbiór testowy (cechy)
    :param y_test_true: Zbiór testowy (prawdziwe etykiety)
    :param title: Tytuł wykresu
    :param class_names: Nazwy klas
    :return: Wyświetlona macierz pomyłek i raport klasyfikacji
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if len(y_test_true.shape) > 1 and y_test_true.shape[1] > 1:
        y_test_true = np.argmax(y_test_true, axis=1)

    cm = confusion_matrix(y_test_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    plt.ylabel("Rzeczywista Klasa")
    plt.xlabel("Przewidziana Klasa")
    plt.title(title)
    plt.show()

    print("\n--- Raport Klasyfikacji ---")
    print(classification_report(y_test_true, y_pred, target_names=class_names))


def run_wheat_seed(epochs=20):
    """
    Trenuje i ocenia model MLP na zbiorze danych Wheat Seed.
    :param epochs: Liczba epok treningu
    :return: Wytrenowany model Keras
    """
    df = pd.read_csv("wheat_seeds_dataset.csv", sep="\s+", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_enc = to_categorical(y_train, 3)
    y_test_enc = to_categorical(y_test, 3)

    model = create_mlp_model(
        input_dim=X_train.shape[1], output_classes=3, hidden_layers=[64, 32]
    )
    model.fit(X_train, y_train_enc, epochs=epochs, batch_size=16, verbose=0)

    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f"[WYNIK] Wheat Seed Accuracy: {acc * 100:.2f}%")
    return model


def run_cifar10(epochs=5):
    """
    Trenuje i ocenia model CNN na zbiorze danych CIFAR-10.
    :param epochs: Liczba epok treningu
    :return: Wytrenowany model Keras, dane testowe (X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    y_train_enc = to_categorical(y_train, 10)
    y_test_enc = to_categorical(y_test, 10)

    model = create_cnn_model(
        input_shape=(32, 32, 3), output_classes=10, filters=[32, 64]
    )

    print("Trenowanie CNN...")
    model.fit(
        X_train,
        y_train_enc,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_test, y_test_enc),
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f"[WYNIK] CIFAR-10 Accuracy: {acc * 100:.2f}%")
    return model, X_test, y_test_enc


def run_fashion_mnist(epochs=5):
    """
    Trenuje i ocenia mały i duży model MLP na zbiorze danych Fashion MNIST.
    :param epochs: Liczba epok treningu
    :return: Model duży Keras, dane testowe (X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train_flat = X_train.reshape(-1, 28 * 28)
    X_test_flat = X_test.reshape(-1, 28 * 28)

    y_train_enc = to_categorical(y_train, 10)
    y_test_enc = to_categorical(y_test, 10)

    print("Trenowanie małego modelu MLP")
    model_small = create_mlp_model(input_dim=784, output_classes=10, hidden_layers=[64])
    model_small.fit(X_train_flat, y_train_enc, epochs=epochs, verbose=0)
    _, acc_small = model_small.evaluate(X_test_flat, y_test_enc, verbose=0)

    print("Trenowanie dużego modelu MLP")
    model_large = create_mlp_model(
        input_dim=784, output_classes=10, hidden_layers=[256, 128, 64]
    )
    model_large.fit(X_train_flat, y_train_enc, epochs=epochs, verbose=0)
    _, acc_large = model_large.evaluate(X_test_flat, y_test_enc, verbose=0)

    print(
        f"[WYNIK] Mały MLP: {acc_small * 100:.2f}% | Duży MLP: {acc_large * 100:.2f}%"
    )

    return model_large, X_test_flat, y_test_enc


if __name__ == "__main__":
    model_wheat = run_wheat_seed()
    model_cifar, X_test_c, y_test_c = run_cifar10(epochs=5)
    model_fashion, X_test_f, y_test_f = run_fashion_mnist(epochs=5)

    fashion_labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    plot_confusion_matrix(
        model_fashion, X_test_f, y_test_f, "Fashion MNIST (Duży MLP)", fashion_labels
    )
