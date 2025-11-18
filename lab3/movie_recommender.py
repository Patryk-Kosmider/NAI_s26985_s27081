"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

System rekomendacji filmów

System ma za zadanie podać 5 rekomedanicji i anyrekomendacji filmowych dla danego użytkownika (użytkownik musi być w bazie danych).
Rekomendacje są wybierane na podstawie wpisów użytkownika do bazy danych i porównywana z wpisami innych użytkowników. Definiowana jest
grupa ankietowanych z podobnymi preferencjami filmowymi i wyświetlane są filmy, których jeszcze nie widzieliśmy. Prócz tego algorytm
wciąga dodatkowe informacje dot. rekomendowanych filmów.
Użytkownik musi podać swoje imię i nazwisko w argumencie --user w formacie: jan_kowalski. Przykładowe wywołanie: pytohn movie_recommender.py --user jan_kowalski.

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install pandas scikit-fuzzy unidecode

"""

import argparse
import requests
import pandas as pd
from sklearn.cluster import KMeans
from prepare_data import prepare_data

API_KEY = "90d0e92c"
def load_data():
    """
    Ładuje i przygotowuje dane do analizy rekomendacji filmów.
    Pivotuje dane do formatu macierzy użytkownik-film.
    :return: Dane w formacie DataFrame oraz pivotowana macierz
    """
    data = prepare_data()
    print(data)
    # Macierz pivotowana użytkownik-fil, np.
    # user | movie1 | movie2 | movie3
    # u1   |  5.0   |  NaN   | 3.0
    # u2   |  NaN   |  4.0   | Na
    pivot = data.pivot_table(index="user", columns="movie", values="rating")
    pivot = pivot.astype(float)
    print(pivot)
    return data, pivot

def cluster_users(pivot, n_clusters=4):
    """
    Grupuje użytkowników na podstawie ich ocen filmów za pomocą KMeans.
    :param pivot: Pivotowana macierz użytkownik-film
    :param n_clusters: Liczba klastrów
    :return: Model KMeans oraz etykiety klastrów dla użytkowników
    """

    # Brakujące oceny wypełniamy zerami - Kmeans nie działa dla brakujących danych
    X = pivot.fillna(0)

    print(X.head())  # pokazujemy kilka wierszy
    print(X.dtypes)  # sprawdzamy typy kolumn
    print(X.isna().sum().sum())  # ile NaN pozostało

    # Tworzenie obiektu AgglomerativeClustering i dopasowanie do danych
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    # Klaster dla każdego użytkownika w formacie DataFrame
    user_cluster = pd.DataFrame({'user': X.index, 'cluster': labels})
    return user_cluster

def get_cluster_movies(df, user_cluster, target_user):
    """
    Pobiera filmy ocenione przez użytkowników z tego samego klastra co użytkownik docelowy.
    :param df: Oryginalne dane
    :param user_cluster: DataFrame z użytkownikami i ich klastrami
    :param target_user: Użytkownik docelowy
    :return: Lista filmów ocenionych przez użytkowników z tego samego klastra
    """
    # Klaster do którego należy użytkownik docelowy
    target_cluster = user_cluster[user_cluster['user'] == target_user]['cluster'].values[0]
    # Pobieramy użytkowników z tego samego klastra i wyciągamy ich oceny filmów
    cluster_users = user_cluster[user_cluster['cluster'] == target_cluster]['user']
    cluster_movies = df[df['user'].isin(cluster_users)]
    return cluster_movies

def recommend_movies(df, user_cluster, target_user, n_recommendations=5):
    """
    Rekomenduje filmy dla użytkownika docelowego na podstawie ocen użytkowników z tego samego klastra.
    :param df: Oryginalne dane
    :param user_cluster: DataFrame z użytkownikami i ich klastrami
    :param target_user: Użytkownik docelowy
    :param n_recommendations: Liczba rekomendacji do zwrócenia
    :return: Lista rekomendowanych filmów
    """
    # Filmy ocenione przez użytkowników z tego samego klastra
    cluster_movies = get_cluster_movies(df, user_cluster, target_user)
    # Filmy które użytkownik już widział
    user_movies = df[df['user'] == target_user]['movie'].tolist()
    # Tworzymy 5 rekomendacji
    # Wybieramy filmy, których nie widział użytkownik, obliczamy średnią ocenę dla każdego filmu w klastrze i sortujemy malejąco (wybieramy top 5 najlepszych)
    recommendations = cluster_movies[~cluster_movies['movie'].isin(user_movies)]
    recommended_movies = recommendations.groupby('movie')['rating'].mean().reset_index()
    recommended_movies = recommended_movies.sort_values(by='rating', ascending=False).head(n_recommendations)

    return recommended_movies['movie'].tolist()

def anti_recommend_movies(df, user_cluster, target_user, n_recommendations=5):
    """
    Anty-rekomenduje filmy dla użytkownika docelowego na podstawie ocen użytkowników z tego samego klastra.
    :param df: Oryginalne dane
    :param user_cluster: DataFrame z użytkownikami i ich klastrami
    :param target_user: Użytkownik docelowy
    :param n_recommendations: Liczba rekomendacji do zwrócenia
    :return: Lista rekomendowanych filmów
    """
    # Filmy ocenione przez użytkowników z tego samego klastra
    cluster_movies = get_cluster_movies(df, user_cluster, target_user)
    # Filmy które użytkownik już widział
    user_movies = df[df['user'] == target_user]['movie'].tolist()
    # Tworzymy 5 rekomendacji
    # Wybieramy filmy, których nie widział użytkownik, obliczamy średnią ocenę dla każdego filmu w klastrze i sortujemy malejąco (wybieramy top 5 najgorszych)
    anti_recommendations = cluster_movies[~cluster_movies['movie'].isin(user_movies)]
    anti_recommended_movies = anti_recommendations.groupby('movie')['rating'].mean().reset_index()
    anti_recommended_movies = anti_recommended_movies.sort_values(by='rating', ascending=True).head(n_recommendations)

    return anti_recommended_movies['movie'].tolist()

def get_movie_info(movie):
    """
    Wyciągnięcie dodatkowych informacji dot. filmu
    :param movie: Tytuł filmu
    :return: Lista dodatkowych informacji o filmie
    """

    url = "http://www.omdbapi.com/"
    params = {
        "t": movie,
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Błąd połączenia", response.text)

    data = response.json()

    if data.get("Response") == "False":
        return {'Title': "None", 'Year': "None", "Genre": "None"}

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--user", type=str, help="Target user for recommendations")
    args = parser.parse_args()
    target_user = args.user

    df, pivot = load_data()
    user_cluster = cluster_users(pivot, n_clusters=5)
    recs = recommend_movies(df, user_cluster, target_user, n_recommendations=5)
    anti_recs = anti_recommend_movies(df, user_cluster, target_user, n_recommendations=5)

    print(f"\nRekomendowane filmy dla użytkownika {target_user}:")
    for movie in recs:
        movie_info = get_movie_info(movie)
        print(f"- {movie}, rok produkcji: {movie_info['Year']}, gatunek: {movie_info['Genre']}")
    print(f"\nAnty-rekomendowane filmy dla użytkownika {target_user}:")
    for movie in anti_recs:
        movie_info = get_movie_info(movie)
        print(f"- {movie}, rok produkcji: {movie_info['Year']}, gatunek: {movie_info['Genre']}")
