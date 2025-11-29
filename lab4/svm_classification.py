import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

WHEAT_DATA = "wheat_seeds_dataset.csv"
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


def prepare_training_data(file_name):
    """
    Przygotowuje dane treningowe i testowe z podanego pliku CSV.
    Wykonuje skalowanie cech i podział na zbiór treningowy/testowy.
    :param file_name: Nazwa pliku CSV (WHEAT_DATA)
    :return: X_train, X_test, y_train, y_test
    """

    data_df = pd.read_csv(file_name, names=COLUMN_NAMES, sep="\s+", engine="python")

    X = data_df.drop("class", axis=1)
    y = data_df["class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(
    X_train, X_test, y_train, y_test, model_description, model
):
    """
    Trenuje i ocenia podany model na danych treningowych i testowych.
    Wyświetla dokładność, raport klasyfikacji oraz macierz pomyłek.
    :param X_train: Dane treningowe (cechy)
    :param X_test: Dane testowe (cechy)
    :param y_train: Etykiety treningowe
    :param y_test: Etykiety testowe
    :param model_description: Opis modelu (np. "SVM z jądrem RBF")
    :param model: Model do trenowania (np. svm.SVC())
    :return: Wytrenowany model
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n--- Ocena modelu: {model_description} na zbiorze {WHEAT_DATA} ---")
    print(f"Dokładność: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(num=f"Macierz Pomyłek - {model_description}", figsize=(7, 7))

    disp.plot(ax=plt.gca(), cmap=plt.cm.Blues)

    plt.title(f"Macierz Pomyłek dla {model_description}")
    plt.show()

    return model


def visualize_data(
    X, y, feature_x, feature_y, features_names, title="Wizualizacja danych"
):
    """
    Wizualizuje dane na wykresie rozrzutu dla dwóch wybranych cech.
    :param X: Dane cech
    :param y: Etykiety klas
    :param feature_x: Indeks cechy dla osi X
    :param feature_y: Indeks cechy dla osi Y
    :param features_names: Nazwy cech
    :param title: Tytuł wykresu
    """

    plt.figure(num="Wizualizacja Danych - Nasiona Pszenicy", figsize=(8, 6))

    scatter = plt.scatter(
        X[:, feature_x], X[:, feature_y], c=y, cmap="viridis", edgecolor="k"
    )
    plt.xlabel(features_names[feature_x])
    plt.ylabel(features_names[feature_y])
    plt.title(title)
    plt.colorbar(scatter, label="Class")
    plt.show()


X_train, X_test, y_train, y_test = prepare_training_data(file_name=WHEAT_DATA)


tree_model = DecisionTreeClassifier(random_state=42)
train_and_evaluate_model(
    X_train, X_test, y_train, y_test, "Drzewo Decyzyjne", tree_model
)

svm_kernels = ["linear", "poly", "rbf", "sigmoid"]
for svm_kernel in svm_kernels:
    model_description = f"SVM (Kernel: {svm_kernel})"
    svm_model = svm.SVC(kernel=svm_kernel, C=1.0, random_state=42)
    train_and_evaluate_model(
        X_train, X_test, y_train, y_test, model_description, svm_model
    )


visualize_data(
    X_train,
    y_train,
    feature_x=0,
    feature_y=1,
    features_names=COLUMN_NAMES[:-1],
    title="Wheat Seeds - Area vs Perimeter",
)
