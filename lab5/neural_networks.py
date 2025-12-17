import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


DEFAULT_EPOCHS = 10
global classes


def prepare_training_data(file_name, result_column, sep):
    """
    Przygotowuje dane treningowe i testowe z podanego pliku CSV.
    Wykonuje skalowanie cech i podział na zbiór treningowy/testowy.
    :param file_name: Nazwa pliku CSV (DATA_FILE)
    :param result_column: Nazwa kolumny z etykietami klas
    :param sep: Separator w pliku CSV
    :return: X_train, X_test, y_train, y_test
    """

    data_df = pd.read_csv(file_name, names=COLUMN_NAMES, sep=sep, engine="python")

    data_df.replace("?", np.nan, inplace=True)
    data_df.dropna(inplace=True)
    y_raw = data_df[result_column]

    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = data_df[result_column].values
        y = y - 1
    
    X = data_df.drop(columns=[result_column], axis=1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_model(input_size, output_classes, hidden_layers=[64, 32]):
    """
    Przygotowanie modelu w pełni połączonej sieci neuronowej (MLP).
    :param input_size: Rozmiar wejścia
    :param output_classes: Klasy wyjściowe
    :param hidden_layers: Warstwa ukryte
    :return: Model Keras
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                hidden_layers[0], activation="relu", input_shape=(input_size,)
            ),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(hidden_layers[1], activation="relu"),
            tf.keras.layers.Dense(output_classes, activation="softmax"),
        ],
        name="model",
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model

def create_cnn_model(input_shape, output_classes):
    """
    Przygotowanie modelu konwolucyjnej sieci neuronowej (CNN).
    :param input_shape: Kształt wejścia
    :param output_classes: Klasy wyjściowe
    :return: Model Keras
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(output_classes, activation="softmax"),
        ],
        name = "cnn_model"
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, epochs=DEFAULT_EPOCHS):
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

    model.save(f"{model.name}_model.keras")

    return model, accuracy, history, loss


if __name__ == "__main__":
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
        if args.model == "wheat_seeds":
            DATA_FILE = "wheat_seeds_dataset.csv"
            COLUMN_NAMES = [
                "area",
                "perimeter",
                "compactness",
                "length_of_kernel",
                "width_of_kernel",
                "asymmetry_coefficient",
                "length_of_groove",
                "class",
            ]

            classes = ["Class 1", "Class 2", "Class 3"]
            result_column = "class"
            sep = "\s+"
            feature_x = 0
            feature_y = 1
        elif args.model == "music_genre":
            # TO DO naprawić formatowanie danych bo to maniana jest
            DATA_FILE = "music_genre.csv"
            COLUMN_NAMES = [
                "instance_id",
                "artist_name",
                "track_name",
                "popularity",
                "acousticness",
                "danceability",
                "duration_ms",
                "energy",
                "instrumentalness",
                "key",
                "liveness",
                "loudness",
                "mode",
                "speechiness",
                "tempo",
                "obtained_date",
                "valence",
                "music_genre",
            ]
            classes = [
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
            ]
            result_column = ["music_genre", "artist_name", "track_name", "popularity", "instance_id", "obtained_date"]
            sep = ","
            feature_x = 0
            feature_y = 1
        else:
            print("Proszę podać poprawny argument --model lub --cnn-model.")
            exit(1)

        X_train, X_test, y_train, y_test = prepare_training_data(
            DATA_FILE, result_column, sep
        )
        train_and_evaluate_model(
            X_train,
            X_test,
            y_train,
            y_test,
            create_model(X_train.shape[1], len(classes)),
        )
    elif args.cnn_model:
        if args.cnn_model == "animals":
            classes = [
                "cat",
                "bird",
                "dog",
                "frog",
                "horse",
                "deer"
            ]
        elif args.cnn_model == "clothes":
            classes = [
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
    else:
        print("Proszę podać poprawny argument --model lub --cnn-model.")
        exit(1)

