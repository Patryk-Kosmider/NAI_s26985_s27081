"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

System rekomendacji filmów (użycie korelacji Pearsona)

System podaje 5 rekomendacji i anty-rekomendacji filmowych dla danego użytkownika.
Rekomendacje są wybierane na podstawie wpisów użytkownika oraz korelacji z innymi użytkownikami.
Użytkownik musi podać swoje imię i nazwisko w argumencie --user w formacie: jan_kowalski.
Przykładowe wywołanie: python movie_recommender.py --user jan_kowalski.

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install pandas unidecode
"""

import argparse
import requests
from prepare_data import prepare_data

API_KEY = "90d0e92c"


def load_data():
    """
    Ładuje dane i pivotuje je do macierzy użytkownik-film
    """
    data = prepare_data()
    pivot = data.pivot_table(index="user", columns="movie", values="rating")
    return data, pivot


def calculate_user_correlation(pivot):
    """
    Oblicza macierz korelacji Pearsona między użytkownikami
    :param pivot: pivotowana macierz użytkownik-film
    :return: DataFrame korelacji użytkownik-użytkownik
    """
    # Pearson ignoruje NaN, więc nie trzeba wypełniać zerami
    user_corr = pivot.T.corr(method="pearson")
    return user_corr


def recommend_movies(df, user_corr, target_user, n_recommendations=5):
    """
    Rekomenduje filmy dla użytkownika docelowego na podstawie ocen użytkowników.
    :param df: Oryginalne dane
    :param user_corr: DataFrame z użytkownikami
    :param target_user: Użytkownik docelowy
    :param n_recommendations: Liczba rekomendacji do zwrócenia
    :return: Lista rekomendowanych filmów
    """
    similar_users = get_similar_users(user_corr, target_user)
    cluster_movies = df[df["user"].isin(similar_users.index)]
    user_movies = df[df["user"] == target_user]["movie"].tolist()

    # Filmy, których target_user nie widział
    recommendations = cluster_movies[~cluster_movies["movie"].isin(user_movies)]

    # Średnia ważona ocena wg korelacji
    recommendations = (
        recommendations.groupby("movie")
        .apply(
            lambda x: (x["rating"] * similar_users[x["user"]].values).sum()
            / similar_users.sum(),
            include_groups=False,
        )
        .reset_index(name="weighted_rating")
    )

    recommendations = recommendations.sort_values(
        by="weighted_rating", ascending=False
    ).head(n_recommendations)
    return recommendations["movie"].tolist()


def anti_recommend_movies(df, user_corr, target_user, n_recommendations=5):
    """
     Anty-rekomenduje filmy dla użytkownika docelowego na podstawie ocen użytkowników.
    :param df: Oryginalne dane
    :param user_corr: DataFrame z użytkownikami
    :param target_user: Użytkownik docelowy
    :param n_recommendations: Liczba rekomendacji do zwrócenia
    :return: Lista rekomendowanych filmów
    """
    similar_users = get_similar_users(user_corr, target_user)
    cluster_movies = df[df["user"].isin(similar_users.index)]
    user_movies = df[df["user"] == target_user]["movie"].tolist()
    # Filmy, których target_user nie widział
    recommendations = cluster_movies[~cluster_movies["movie"].isin(user_movies)]
    # Średnia ważona ocena wg korelacji
    recommendations = (
        recommendations.groupby("movie")
        .apply(
            lambda x: (x["rating"] * similar_users[x["user"]].values).sum()
            / similar_users.sum(),
            include_groups=False,
        )
        .reset_index(name="weighted_rating")
    )
    recommendations = recommendations.sort_values(
        by="weighted_rating", ascending=True
    ).head(n_recommendations)
    return recommendations["movie"].tolist()


def get_movie_info(movie):
    """
    Wyciągnięcie dodatkowych informacji dot. filmu
    :param movie: Tytuł filmu
    :return: Lista dodatkowych informacji o filmie
    """
    url = "http://www.omdbapi.com/"
    params = {"t": movie, "apikey": API_KEY}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Błąd połączenia", response.text)
        return {"Title": "None", "Year": "None", "Genre": "None"}

    data = response.json()
    if data.get("Response") == "False":
        return {"Title": "None", "Year": "None", "Genre": "None"}
    return data


def get_similar_users(user_corr, target_user, top_n=5):
    """
    Pobiera top N użytkowników najbardziej podobnych do target_user
    i wyświetla ich w procentach.
    :param user_corr: DataFrame z użytkownikami
    :param target_user: Użytkownik docelowy
    :param top_n: Ilość najlepszych wyników jaką chcemy
    :return: Series podobnych userów
    """
    similar_users = (
        user_corr[target_user]
        .sort_values(ascending=False)
        .drop(target_user)
        .head(top_n)
    )

    return similar_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="System rekomendacji filmów (użycie korelacji Pearsona)"
    )
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="Podaj użtkownika (format: imie_nazwisko",
    )
    args = parser.parse_args()

    target_user = args.user
    df, pivot = load_data()

    user_corr = calculate_user_correlation(pivot)

    recs = recommend_movies(df, user_corr, target_user)
    anti_recs = anti_recommend_movies(df, user_corr, target_user)
    similar_users = get_similar_users(user_corr, target_user)

    print(f"\nRekomendowane filmy dla użytkownika {target_user}:")
    for movie in recs:
        info = get_movie_info(movie)
        print(f"- {movie}, rok produkcji: {info['Year']}, gatunek: {info['Genre']}")

    print(f"\nAnty-rekomendowane filmy dla użytkownika {target_user}:")
    for movie in anti_recs:
        info = get_movie_info(movie)
        print(f"- {movie}, rok produkcji: {info['Year']}, gatunek: {info['Genre']}")
    print(f"\nPodobni użytkownicy do {target_user}")
    for user, corr in similar_users.items():
        print(f"- {user}: {corr * 100:.2f}% podobieństwa")
