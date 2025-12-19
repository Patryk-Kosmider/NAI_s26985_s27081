import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def prepare_wheat_data(file_name, result_column):
    """
    Przygotowuje dane treningowe i testowe z podanego pliku CSV.
    Wykonuje skalowanie cech i podział na zbiór treningowy/testowy.
    :param file_name: Nazwa pliku CSV (DATA_FILE)
    :param result_column: Nazwa kolumny z etykietami klas
    :param COLUMN_NAMES: Nazwy kolumn w pliku CSV
    :return: X_train, X_test, y_train, y_test
    """
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
    data_df = pd.read_csv(file_name, names=COLUMN_NAMES, sep="\s+", engine="python")

    y = data_df[result_column].values
    y = y - 1

    X = data_df.drop(columns=[result_column], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def prepare_music_data(file_name, result_column):
    """
    Przygotowuje dane treningowe i testowe z podanego pliku CSV.
    Wykonuje skalowanie cech i podział na zbiór treningowy/testowy.
    :param file_name: Nazwa pliku CSV (DATA_FILE)
    :param result_column: Nazwa kolumny z etykietami klas
    :param sep: Separator w pliku CSV
    :return: X_train, X_test, y_train, y_test
    """
    data_df = pd.read_csv(file_name, sep=",", on_bad_lines="skip")
    cols_to_drop = [
        "instance_id",
        "artist_name",
        "track_name",
        "popularity",
        "obtained_date",
    ]
    data_df = data_df.drop(
        columns=[c for c in cols_to_drop if c in data_df.columns], axis=1
    )

    data_df = data_df.replace("?", np.nan)
    data_df = data_df.dropna()
    if "mode" in data_df.columns:
        data_df["mode"] = data_df["mode"].map({"Major": 1, "Minor": 0})

    if "key" in data_df.columns:
        encoder = LabelEncoder()
        data_df["key"] = encoder.fit_transform(data_df["key"])

    features = data_df.columns.drop(result_column)
    for col in features:
        data_df[col] = pd.to_numeric(data_df[col])

    le_genre = LabelEncoder()
    y = le_genre.fit_transform(data_df[result_column].values)

    X = data_df.drop(columns=[result_column], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def prepare_fashion_data(dataset, img_height, img_width):
    """
    Przygotowuje dane treningowe i testowe dla CNN.
    Normalizuje obrazy i dzieli na zbiory treningowe/testowe.
    :param dataset: Zbiór danych (np. Fashion MNIST)
    :param img_height: Wysokość obrazu
    :param img_width: Szerokość obrazu
    :return: X_train, X_test, y_train, y_test
    """
    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    X_train = X_train.reshape(-1, img_height, img_width, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, img_height, img_width, 1).astype("float32") / 255.0

    return X_train, X_test, y_train, y_test


def prepare_animal_data(dataset, img_height, img_width):
    """
    Przygotowuje dane treningowe i testowe dla CNN.
    Normalizuje obrazy i dzieli na zbiory treningowe/testowe.
    :param dataset: Zbiór danych (np. Fashion MNIST)
    :param img_height: Wysokość obrazu
    :param img_width: Szerokość obrazu
    :return: X_train, X_test, y_train, y_test
    """

    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    animal_map = {
        2: 0,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
    }

    train_mask = np.isin(y_train, list(animal_map.keys()))
    test_mask = np.isin(y_test, list(animal_map.keys()))

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    y_train = np.array([animal_map[y] for y in y_train])
    y_test = np.array([animal_map[y] for y in y_test])

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return X_train, X_test, y_train, y_test
