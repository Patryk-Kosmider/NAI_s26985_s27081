from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
import math
# TODO: DONE? Trzeba spisac autorow, zasady, instrukcje przygotowania do uruchomienia

"""
Autorzy:
- Patryk Kosmider
- Ziemowit Orlikowski

Cel:
Celem gry jest doprowadzenie pilki na do bramki(x) przeciwnika.
Gracz moze zwyciezyc na dwa sposoby:
- strzelenie bramki przeciwnikowi
- zablokowanie przeciwnika tak ze nie ma mozliwosci wykonania zadnego ruchu

Zasady:
- Kazdy gracz wykonuje jeden ruch na ture w dowolnym kierunku (lewy gorny, gorny, prawy gorny, lewo, prawo, lewy dolny, prawy dolny, dol)
- Nie mozna stawac na polu na ktorym juz byla pilka

Przygotowanie do uruchomienia - wymagania:

Srodowisko z Pythonem (Testowne na wersji 3.12)
Instalacja pakietu easyAI:

- pip install easyAI
"""

# TODO: DONE Ogarnąć strzelanie goli - gol tylko jeśli piła jest na środku pierwszego/ostatniego rzędu - to dziala, ale X jest nad kropka oznaczajaca bramke. Moze jakos inaczej sformulowac ta tabele? By wygladala bardziej jak na kurniku xd.

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

CHARS_MAPPING = {
    "q": "\\",  # lewy górny
    "w": "|",  # góra
    "e": "/",  # prawy górny
    "a": "-",  # lewo
    "d": "-",  # prawo
    "z": "/",  # lewy dolny
    "x": "|",  # dół
    "c": "\\",  # prawy dolny
}

#TODO: DONE Słownik z ruchami, ale mapowanie na rysowanie linii - np. '\': (-1, -1), albo q: '\', zależy jak lepiej.

class PaperSoccer(TwoPlayerGame):
    def __init__(self, players):
        """
        Inicjalizuje grę Paper Soccer z domyślnymi ustawieniami.
        :param players: Lista graczy biorących udział w grze.
        :return: None
        """
        self.players = players
        self.rows = 9
        self.cols = 7
        self.ball_start = (self.rows // 2, self.cols // 2)
        self.ball = self.ball_start
        self.current_player = 1
        self.goal_top = (0, self.cols // 2)
        self.goal_bottom = (self.rows - 1, self.cols // 2)
        self.moves = []
        self.char_list = []
        self.blocked_position = []

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
        move_char = CHARS_MAPPING[move]
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

        if (new_r == 0 or new_r == self.rows - 1) and new_c != self.cols // 2:
            print("Nie możesz wejść do bramki z boku! Tylko środek (x) jest otwarty.")
            self.show()
            return

        self.moves.append(moving)
        self.char_list.append((start, move_char))
        self.blocked_position.append(start)
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

            # Sprawdzenie bramki
            if (new_r == 0 or new_r == self.rows - 1) and new_c != self.cols // 2:
                continue
            # Sprawdzenie granic boiska
            if new_r < 0 or new_r >= self.rows or new_c < 0 or new_c >= self.cols:
                continue
            # Sprawdzanie czy dane miejsce bylo juz uzyte
            if end in self.blocked_position:
                continue

            moves.append(move)
        return moves

    def scoring(self):
        # TODO: DONE? Heurestyka dla AI, jaki ruch ma wybrac, np. odleglosc do bramki przeciwnika. Wazne zeby ogarnac ze na bramke przeciwnika ma isc xd
        br, bc = self.ball
        # euklides do gornej
        dist_to_bottom_goal = math.sqrt((br - (self.goal_bottom[0] - 1))**2 + (bc - (self.goal_bottom[1] // 2))**2)
        # euklides do dolnej
        dist_to_top_goal = math.sqrt((br - (self.goal_top[0] + 1))**2 + (bc - (self.goal_top[1] // 2))**2)

        return dist_to_bottom_goal - dist_to_top_goal

    def show(self):
        """
        """
        field_width = self.cols * 2 + 1
        pad = " " * 2
        goal = "|x|"
        goal_line = pad + "-" * (field_width // 2 - 1) + goal + "-" * (field_width // 2 - 1)
        # TODO: DONE Rysowanie linii - wykonanych ruchow, najlepiej z '-', '|', '/' i '\'. W tablicy move maja byc zapisane ruchy wiec trzeba je przejsc i narysowac linie.
        board = []
        for i in range(self.rows):
            row = []
            for _ in range(self.cols):
                row.append(".")
            board.append(row)

        for (start, char) in self.char_list:
            r1, c1 = start
            board[r1][c1] = char

        br, bc = self.ball
        board[br][bc] = "o"

        for r in range(self.rows):
            if r == 0 or r == self.rows-1:
                print(goal_line)
                continue
            print(pad + "|" + " ".join(board[r]) + "|")

if __name__ == "__main__":

    game = PaperSoccer([Human_Player(), AI_Player(Negamax(3))])
    while True:
        # Zamiast game.play() robimy pętlę, żeby móc wyświetlać planszę i komunikaty
        game.show()
        moves = game.possible_moves()
        if game.ball == game.goal_top:
            print("Wygrał gracz!")
            break
        elif game.ball == game.goal_bottom:
            print("Wygrało AI!")
            break
        elif not moves:
            print("Brak możliwych ruchów! Wygrał gracz", 3 - game.current_player, "!")
            break
        print("Możliwe ruchy:", moves)
        if game.current_player == 1:
            move = input("Player 1 what do you play? ")
        else:
            move = game.players[1].ask_move(game)
            print(f"AI wybrało ruch: {move}")
        game.make_move(move)
