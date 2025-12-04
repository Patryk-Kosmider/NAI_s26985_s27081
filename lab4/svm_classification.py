"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

Klasyfikacja danych za pomocą SVM i Drzewa Decyzyjnego

System trenuje i ocenia modele klasyfikacji SVM z różnymi jądrami i parametrami oraz drzewo decyzyjne. Do klasyfikacji dostępne są
dwa zbiory danych: wheat_seeds_dataset.csv - https://archive.ics.uci.edu/dataset/236/seeds oraz apple_quality.csv - https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality.

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install pandas numpy matplotlib scikit-learn argparse

Przykładowe uruchomienie:
  python svm_classification.py --datafile wheat_seeds_dataset.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
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

global classes


def prepare_training_data(file_name, result_column="class", sep="\s+"):
    """
    Przygotowuje dane treningowe i testowe z podanego pliku CSV.
    Wykonuje skalowanie cech i podział na zbiór treningowy/testowy.
    :param file_name: Nazwa pliku CSV (DATA_FILE)
    :param result_column: Nazwa kolumny z etykietami klas
    :param sep: Separator w pliku CSV
    :return: X_train, X_test, y_train, y_test
    """

    data_df = pd.read_csv(file_name, names=COLUMN_NAMES, sep=sep, engine="python")

    if result_column == "Quality":
        mapping = {"good": 0, "bad": 1}
        y = data_df[result_column].map(mapping).values
    else:
        y = data_df[result_column].values

    X = data_df.drop(result_column, axis=1)

    if "A_id" in X.columns:
        X = X.drop("A_id", axis=1)

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

    print(f"\n--- Ocena modelu: {model_description} na zbiorze {DATA_FILE} ---")
    print(f"Dokładność: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

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

    plt.figure(num="Wizualizacja Danych", figsize=(8, 6))

    scatter = plt.scatter(
        X[:, feature_x], X[:, feature_y], c=y, cmap="viridis", edgecolor="k"
    )
    plt.xlabel(features_names[feature_x])
    plt.ylabel(features_names[feature_y])
    plt.title(title)
    plt.colorbar(scatter, label="Class")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Klasyfikacja danych za pomocą SVM i Drzewa Decyzyjnego"
    )
    parser.add_argument(
        "--datafile",
        type=str,
        required=True,
        help="Podaj nazwę pliku z danymi (wheat_seeds_dataset.csv lub diabetes_prediction_dataset.csv)",
    )
    args = parser.parse_args()

    DATA_FILE = args.datafile

    if DATA_FILE == "wheat_seeds_dataset.csv":
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
    elif DATA_FILE == "apple_quality.csv":
        COLUMN_NAMES = [
            "A_id",
            "Size",
            "Weight",
            "Sweetness",
            "Crunchiness",
            "Juiciness",
            "Ripeness",
            "Acidity",
            "Quality",
        ]
        classes = ["Good", "Bad"]
        result_column = "Quality"
        sep = ","
        feature_x = 3
        feature_y = 4
    else:
        print(
            "Nieobsługiwany plik danych. Użyj wheat_seeds_dataset.csv lub apple_quality.csv."
        )
        exit(1)

    X_train, X_test, y_train, y_test = prepare_training_data(
        file_name=DATA_FILE, result_column=result_column, sep=sep
    )

    tree_model = DecisionTreeClassifier(random_state=42)
    train_and_evaluate_model(
        X_train, X_test, y_train, y_test, "Drzewo Decyzyjne", tree_model
    )

    svm_experiments = [
        {"kernel": "linear", "C": 1.0, "model_description": "SVM (Linear, C=1.0 - Bazowe)"},
        {"kernel": "linear", "C": 0.01, "model_description": "SVM (Linear, C=0.01 - Duża regularyzacja)"},

        {"kernel": "poly", "C": 1.0, "degree": 3, "model_description": "SVM (Poly, degree=3 - Bazowe)"},
        {"kernel": "poly", "C": 1.0, "degree": 5, "model_description": "SVM (Poly, degree=5 - Większa złożoność)"},

        {"kernel": "rbf", "C": 1.0, "gamma": 'scale', "model_description": "SVM (RBF, gamma=scale - Bazowe)"},
        {"kernel": "rbf", "C": 1.0, "gamma": 0.01, "model_description": "SVM (RBF, gamma=0.01 - Gładka granica)"},

        {"kernel": "sigmoid", "C": 1.0, "model_description": "SVM (Sigmoid, C=1.0 - Bazowe)"},
        {"kernel": "sigmoid", "C": 10.0, "model_description": "SVM (Sigmoid, C=10.0 - Mniejsza regularyzacja)"},
    ]

    for svm_exp in svm_experiments:
        svm_model = svm.SVC(
            kernel=svm_exp["kernel"],
            C=svm_exp['C'],
            gamma=svm_exp.get('gamma', 'scale'),
            degree=svm_exp.get('degree', 3),
            random_state = 42,
            class_weight="balanced",
        )

        train_and_evaluate_model(X_train, X_test, y_train, y_test, svm_exp['model_description'], svm_model)

    visualize_data(
        X_train,
        y_train,
        feature_x=feature_x,
        feature_y=feature_y,
        features_names=COLUMN_NAMES[:-1],
        title="Wizualizacja danych treningowych",
    )
