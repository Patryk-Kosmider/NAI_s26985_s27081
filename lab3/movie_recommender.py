import argparse

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from prepare_data import prepare_data


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

def cluster_users(pivot, n_clusters=6):
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
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    labels = agglo.fit_predict(X)

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

def cluster_user_info(pivot, user_cluster, target_user):
    """
    Wyświetla informacje o klastrze użytkownika docelowego oraz podobieństwo do innych użytkowników w klastrze.
    :param pivot: Standardowa macierz użytkownik-film
    :param user_cluster: Klastery użytkowników
    :param target_user: Klaster docelowy
    :return: None
    """
    print(user_cluster['user'].tolist())
    print("Szukany użytkownik:", target_user)

    target_cluster = user_cluster[user_cluster['user'] == target_user]['cluster'].values[0]
    cluster_members = user_cluster[user_cluster['cluster'] == target_cluster]['user'].tolist()
    cluster_members.remove(target_user)

    print(f"Użytkownik {target_user} należy do klastra {target_cluster}, z którego członkami są:) {cluster_members}")

    corr_matrix = pivot.T.corr(method='pearson')
    similarity_df = pd.DataFrame({
        'user': cluster_members,
        'similarity_percent': (corr_matrix.loc[target_user, cluster_members] * 100).round(2)
    }).sort_values(by='similarity_percent', ascending=False)

    print("\nPodobieństwo do pozostałych użytkowników w klastrze:")
    for _, row in similarity_df.iterrows():
        print(f"- {row['user']}: {row['similarity_percent']}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--user", type=str, help="Target user for recommendations")
    args = parser.parse_args()
    target_user = args.user

    df, pivot = load_data()
    user_cluster = cluster_users(pivot, n_clusters=5)
    cluster_user_info(pivot, user_cluster, target_user)
    recs = recommend_movies(df, user_cluster, target_user, n_recommendations=5)
    anti_recs = anti_recommend_movies(df, user_cluster, target_user, n_recommendations=5)

    print(f"\nRekomendowane filmy dla użytkownika {target_user}:")
    for movie in recs:
        print(f"- {movie}")
    print(f"\nAnty-rekomendowane filmy dla użytkownika {target_user}:")
    for movie in anti_recs:
        print(f"- {movie}")
