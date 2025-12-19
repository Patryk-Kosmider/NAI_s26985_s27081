"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

Sieci neuronowe dla klasyfikacji danych

System pozwala tworzyć dwa rodzaje modeli: do klasyfikacji danych oraz do klasyfikacji zdjęć. Do klasyfikacji danych dostępne są
dwa zbiory danych: wheat_seeds_dataset.csv - https://archive.ics.uci.edu/dataset/236/seeds oraz music_genre.csv - https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre.
Klasyfikacja zdjęć odbywa się na podstawie zbiorów danych CIFAR-10 (zwierzęta) oraz Fashion MNIST (odzież). Modele są zapisywane lokalnie na dysku w folderze roboczym.

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install pandas numpy matplotlib scikit-learn argparse, tensorflow

Przykładowe uruchomienia:
  python neural_networks.py --model wheat_seeds
  python neural_networks.py --model music_genre
  python neural_networks.py --cnn-model animals
  python neural_networks.py --cnn-model clothes
# Dla identyfikacji obrazu
  python identify_image.py --image cat.jpg --model animals_model.keras --type animals
  python identify_image.py --image t-shirt.jpg --model clothes_model.keras --type clothes
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from prepare_data import (
    prepare_wheat_data,
    prepare_fashion_data,
    prepare_animal_data,
    prepare_music_data,
)
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

DEFAULT_EPOCHS = 10


def create_model(
    input_size, output_classes, dense_units=[64, 32], model_name="model"
):
    """
    Przygotowanie modelu w pełni połączonej sieci neuronowej (MLP).
    :param input_size: Rozmiar wejścia
    :param output_classes: Klasy wyjściowe
    :param dense_units: Ilość neuronów w warstwach ukrytych
    :return: Model Keras
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dense_units[0], activation="relu", input_shape=(input_size,)
            ),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(dense_units[1], activation="relu"),
            tf.keras.layers.Dense(output_classes, activation="softmax"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_model_two(
    input_size, output_classes, dense_units=[64, 32, 16], model_name="bigger_model"
):
    """
    Przygotowanie modelu w pełni połączonej sieci neuronowej (MLP).
    :param input_size: Rozmiar wejścia
    :param output_classes: Klasy wyjściowe
    :param dense_units: Ilość neuronów w warstwach ukrytych
    :return: Model Keras
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dense_units[0], activation="relu", input_shape=(input_size,)
            ),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(dense_units[1], activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(dense_units[2], activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_classes, activation="softmax"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_cnn_model(input_shape, output_classes, model_name="cnn_model"):
    """
    Przygotowanie modelu konwolucyjnej sieci neuronowej (CNN).
    :param input_shape: Kształt wejścia
    :param output_classes: Klasy wyjściowe
    :return: Model Keras
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(output_classes, activation="softmax"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_and_evaluate_model(
    X_train, X_test, y_train, y_test, model, classes, epochs=DEFAULT_EPOCHS
):
    """
    Trenuje i ocenia podany model na danych treningowych i testowych.
    Wyświetla dokładność, raport klasyfikacji oraz macierz pomyłek.
    :param X_train: Dane treningowe (cechy)
    :param X_test: Dane testowe (cechy)
    :param y_train: Etykiety treningowe
    :param y_test: Etykiety testowe
    :param model: Model do trenowania
    :return: Wytrenowany model, dokładność, historia treningu, strata
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(f"Dokładność: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure(num=f"Macierz Pomyłek", figsize=(7, 7))

    disp.plot(ax=plt.gca(), cmap=plt.cm.Blues)

    plt.title(f"Macierz Pomyłek")
    plt.show()

    model.save(f"{model.name}.keras")

    return model, accuracy, history, loss


def main():
    parser = argparse.ArgumentParser(
        description="Trenowanie i zapisywanie modeli neuronowych"
    )
    parser.add_argument(
        "--model",
        choices=["wheat_seeds", "music_genre"],
        required=False,
        help="Podaj dataset do trenowania modelu (wheat_seeds lub music_genre)",
    )
    parser.add_argument(
        "--cnn-model",
        choices=["animals", "clothes"],
        required=False,
        help="Podaj dataset do trenowania modelu CNN (animals lub clothes)",
    )
    args = parser.parse_args()

    if args.model:
        datasets = {
            "wheat_seeds": {
                "file": "wheat_seeds_dataset.csv",
                "classes": ["Class 1", "Class 2", "Class 3"],
                "result_column": "class",
                "model_name": "wheat_seeds_model",
                "prepare_fn": prepare_wheat_data,
                "models_fn": [create_model, create_model_two],
            },
            "music_genre": {
                "file": "music_genre.csv",
                "classes": [
                    "Electronic",
                    "Anime",
                    "Jazz",
                    "Alternative",
                    "Country",
                    "Rap",
                    "Blues",
                    "Rock",
                    "Classical",
                    "Hip-Hop",
                ],
                "result_column": "music_genre",
                "model_name": "music_genre_model",
                "prepare_fn": prepare_music_data,
                "models_fn": [create_model],
            },
        }
        config = datasets[args.model]
        classes = config["classes"]
        X_train, X_test, y_train, y_test = config["prepare_fn"](
            config["file"], config["result_column"]
        )
        for model_fn in config["models_fn"]:
            model = model_fn(
                X_train.shape[1],
                len(config["classes"]),
                model_name=config["model_name"],
            )
            train_and_evaluate_model(X_train, X_test, y_train, y_test, model, classes)

    elif args.cnn_model:
        cnn_datasets = {
            "animals": {
                "classes": ["bird", "cat", "deer", "dog", "frog", "horse"],
                "dataset": tf.keras.datasets.cifar10,
                "img_height": 32,
                "img_width": 32,
                "model_name": "animals_model",
                "prepare_fn": prepare_animal_data,
            },
            "clothes": {
                "classes": [
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
                ],
                "dataset": tf.keras.datasets.fashion_mnist,
                "img_height": 28,
                "img_width": 28,
                "model_name": "clothes_model",
                "prepare_fn": prepare_fashion_data,
            },
        }
        config = cnn_datasets[args.cnn_model]
        classes = config["classes"]
        X_train, X_test, y_train, y_test = config["prepare_fn"](
            config["dataset"], config["img_height"], config["img_width"]
        )
        model = create_cnn_model(
            X_train.shape[1:], len(config["classes"]), model_name=config["model_name"]
        )
        train_and_evaluate_model(X_train, X_test, y_train, y_test, model, classes)

    else:
        print("Proszę podać poprawny argument --model lub --cnn-model.")
        exit(1)


if __name__ == "__main__":
    main()
