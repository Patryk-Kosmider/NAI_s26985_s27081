from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

# TODO: Trzeba spisac autorow, zasady, instrukcje przygotowania do uruchomienia

# TODO: Ogarnąć strzelanie goli - gol tylko jeśli piła jest na środku pierwszego/ostatniego rzędu - to dziala, ale X jest nad kropka oznaczajaca bramke. Moze jakos inaczej sformulowac ta tabele? By wygladala bardziej jak na kurniku xd.

DIRECTIONS = {
    "q": (-1, -1),  # lewy górny
    "w": (-1, 0),   # góra
    "e": (-1, 1),   # prawy górny
    "a": (0, -1),   # lewo
    "d": (0, 1),    # prawo
    "z": (1, -1),   # lewy dolny
    "x": (1, 0),    # dół
    "c": (1, 1),    # prawy dolny
}

#TODO: Słownik z ruchami, ale mapowanie na rysowanie linii - np. '\': (-1, -1), albo q: '\', zależy jak lepiej.

class PaperSoccer(TwoPlayerGame):
    def __init__(self, players):
        """
        Inicjalizuje grę Paper Soccer z domyślnymi ustawieniami.
        :param players: Lista graczy biorących udział w grze.
        :return: None
        """
        self.players = players
        self.rows = 7
        self.cols = 7
        self.ball_start = (self.rows // 2, self.cols // 2)
        self.ball = self.ball_start
        self.current_player = 1
        self.goal_top = (0, self.cols // 2)
        self.goal_bottom = (self.rows - 1, self.cols // 2)
        self.moves = []

    def is_over(self):
        if self.ball == self.goal_top:
            return True
        if self.ball == self.goal_bottom:
            return True
        if not self.possible_moves():
            return True
        return False

    def make_move(self, move):
        """
        Wykonuje ruch piłką w wybranym kierunku, zapisuje ruch i sprawdza czy ruch jest dozwolony.
        :param move: Kierunek ruchu (np. 'w', 'a', 'd', itp.)
        :return: None
        """
        if move not in DIRECTIONS:
            print("Nieprawidłowy kierunek ruchu. Dostępne ruchy:", list(DIRECTIONS.keys()))
            self.show()
            return
        dr, dc = DIRECTIONS[move]
        new_r = self.ball[0] + dr
        new_c = self.ball[1] + dc
        start = self.ball
        end = (new_r, new_c)
        moving = (start, end)

        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            print("Ruch poza planszą! Nie możesz wykonać tego ruchu.")
            print("Możliwe ruchy:", self.possible_moves())
            self.show()
            return

        if moving in self.moves or (end, start) in self.moves:
            print("Ten ruch już był! Nie możesz wykonać tej samej trasy.")
            print("Możliwe ruchy:", self.possible_moves())
            self.show()
            return

        self.moves.append(moving)
        self.ball = end
        self.current_player = 3 - self.current_player

    def possible_moves(self):
        """
        Zwraca listę możliwych ruchów z aktualnej pozycji piłki.
        :return: Lista dozwolonych kierunków ruchu
        """
        moves = []
        for move, (dr, dc) in DIRECTIONS.items():
            new_r = self.ball[0] + dr
            new_c = self.ball[1] + dc
            start = self.ball
            end = (new_r, new_c)
            moving = (start, end)
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                if moving not in self.moves and (end, start) not in self.moves:
                    moves.append(move)
        return moves

    def scoring(self):
        # TODO: Heurestyka dla AI, jaki ruch ma wybrac, np. odleglosc do bramki przeciwnika. Wazne zeby ogarnac ze na bramke przeciwnika ma isc xd
        pass

    def show(self):
        """
        """
        field_width = self.cols * 2 + 1
        pad = " " * 2

        goal_line = pad + " " * (field_width // 2) + "x"

        # TODO: Rysowanie linii - wykonanych ruchow, najlepiej z '-', '|', '/' i '\'. W tablicy move maja byc zapisane ruchy wiec trzeba je przejsc i narysowac linie.

        print(goal_line)
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.ball:
                    row.append("o")
                else:
                    row.append(".")
            print(pad + "|" + " ".join(row) + "|")
        print(goal_line)

if __name__ == "__main__":

    game = PaperSoccer([Human_Player(), AI_Player(Negamax(3))])
    game.play()

    if game.ball == game.goal_top:
        print("Wygrał gracz!")
    elif game.ball == game.goal_bottom:
        print("Wygrało AI!")
    else:
        winner = 3 - game.current_player
        print(f"Brak możliwych ruchów! Wygrał gracz {winner}!")
