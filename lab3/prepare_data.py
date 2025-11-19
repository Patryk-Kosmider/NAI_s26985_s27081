import pandas as pd
import unidecode

def prepare_data():
    """
    Formatowanie danych z pliku CSV do postaci odpowiedniej do analizy - [user, movie, rating]
    :return: Plik CSV z sformatowanymi danymi
    """
    raw_data = "dane.csv"
    clean_data = "formatted_data.csv"
    df = pd.read_csv(raw_data, encoding="utf-8", header=None)
    data = []

    for idx, row in df.iterrows():
        user = unidecode.unidecode(row[0].lower().replace(" ", "_"))
        movies_ratings = row[1:].dropna().tolist()  # usuwamy puste kolumny

        # iteracja po parach movie + rating
        i = 0
        while i < len(movies_ratings) - 1:
            movie = str(movies_ratings[i]).strip()
            rating = movies_ratings[i + 1]
            try:
                rating_val = float(rating)
                data.append([user, movie, rating_val])
                i += 2
            except:
                # jeśli coś nie jest liczbą, przesuwamy się o 1 i próbujemy dalej
                i += 1

    new_df = pd.DataFrame(data, columns=["user", "movie", "rating"])
    new_df.to_csv(clean_data, index=False, encoding="utf-8")
    return new_df
if __name__ == "__main__":
    prepare_data()
