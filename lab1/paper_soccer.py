import math

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

Cel:
Celem gry jest doprowadzenie piłki do bramki (|x|) przeciwnika.
Gracz może zwyciężyć na dwa sposoby:
- strzelenie bramki przeciwnikowi
- zablokowanie przeciwnika tak, że nie ma on możliwości wykonania żadnego ruchu

Zasady:
- Każdy gracz wykonuje jeden ruch na turę w dowolnym kierunku (lewy górny, górny, prawy górny, lewy, prawy, lewy dolny, prawy dolny, dolny)
- Nie można stawać na polu, na którym już była piłka.

Przygotowanie do uruchomienia - wymagania:

Środowisko z Pythonem
Instalacja pakietu easyAI:

- pip install easyAI
"""

DIRECTIONS = {
    "q": (-1, -1),  # lewy górny
    "w": (-1, 0),  # góra
    "e": (-1, 1),  # prawy górny
    "a": (0, -1),  # lewo
    "d": (0, 1),  # prawo
    "z": (1, -1),  # lewy dolny
    "x": (1, 0),  # dół
    "c": (1, 1),  # prawy dolny
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
        """
        Sprawdza czy warunki zakonczenia gry zostały spełnione
        :return: True/False
        """
        if self.ball == self.goal_top:
            print("Wygrał gracz!")
            return True
        if self.ball == self.goal_bottom:
            print("Wygrało AI!")
            return True
        if not self.possible_moves():
            print("Brak możliwych ruchów! Wygrał gracz", 3 - game.current_player, "!")
            return True
        return False

    def make_move(self, character):
        """
        Wykonuje ruch piłką w wybranym kierunku, zapisuje ruch i sprawdza czy ruch jest dozwolony.
        :param character: Kierunek ruchu (np. 'w', 'a', 'd', itp.)
        :return: None
        """
        if character not in DIRECTIONS:
            print(
                "Nieprawidłowy kierunek ruchu. Dostępne ruchy:", list(DIRECTIONS.keys())
            )
            return
        dr, dc = DIRECTIONS[character]
        move_char = CHARS_MAPPING[character]
        new_r = self.ball[0] + dr
        new_c = self.ball[1] + dc
        start = self.ball
        # nowa pozycja piłki po ruchu
        end = (new_r, new_c)
        # ruch jako para współrzędnych
        moving = (start, end)

        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            print("Ruch poza planszą! Nie możesz wykonać tego ruchu.")
            print("Możliwe ruchy:", self.possible_moves())
            return

        if moving in self.moves or (end, start) in self.moves:
            print("Ten ruch już był! Nie możesz wykonać tej samej trasy.")
            print("Możliwe ruchy:", self.possible_moves())
            return

        if (new_r == 0 or new_r == self.rows - 1) and new_c != self.cols // 2:
            print("Nie możesz wejść do bramki z boku! Tylko środek (x) jest otwarty.")
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
        moves_list = []
        for character, (dr, dc) in DIRECTIONS.items():
            new_r = self.ball[0] + dr
            new_c = self.ball[1] + dc
            end = (new_r, new_c)

            # Sprawdzenie bramki
            if (new_r == 0 or new_r == self.rows - 1) and new_c != self.cols // 2:
                continue
            # Sprawdzenie granic boiska
            if new_r < 0 or new_r >= self.rows or new_c < 0 or new_c >= self.cols:
                continue
            # Sprawdzanie czy dane miejsce bylo juz uzyte
            if end in self.blocked_position:
                continue

            moves_list.append(character)
        return moves_list

    def scoring(self):
        """
        Logika punktacji dla AI, wyliczanie odległości piłki do obu bramek
        :return: Różnica odległości piłki od dolnej bramki do górnej
        """
        # pozycja piłki
        br, bc = self.ball
        # euklides do górnej
        dist_to_bottom_goal = math.sqrt(
            (br - (self.goal_bottom[0])) ** 2 + (bc - (self.goal_bottom[1])) ** 2
        )
        # euklides do dolnej
        dist_to_top_goal = math.sqrt(
            (br - (self.goal_top[0])) ** 2 + (bc - (self.goal_top[1])) ** 2
        )
        # Heurystyka:
        # Zwracamy różnicę: (odległość do dolnej) - (odległość do górnej)
        # Dzięki temu przykładowo:
        # - piłka w pozycji bliżej AI (górna bramka) - scoring ujemny - AI gra w dół by maksymalizować
        # - piłka w pozycji bliżej gracza (dolna bramka) - scoring dodatni - AI unika góry
        # - piłka na środku - scoring bliski 0 - AI nie faworyzuje żadnej strony
        return dist_to_bottom_goal - dist_to_top_goal

    def show(self):
        """
        Generowanie planszy, rysowanie boiska, piłki oraz ruchów graczy
        :return: None
        """
        field_width = self.cols * 2 + 1
        pad = " " * 2
        goal = "|x|"
        goal_line = (
                pad + "-" * (field_width // 2 - 1) + goal + "-" * (field_width // 2 - 1)
        )
        board = []
        for _ in range(self.rows):
            row = []
            for _ in range(self.cols):
                row.append(".")
            board.append(row)

        for start, char in self.char_list:
            r1, c1 = start
            board[r1][c1] = char

        br, bc = self.ball
        board[br][bc] = "o"

        for r in range(self.rows):
            if r == 0 or r == self.rows - 1:
                print(goal_line)
                continue
            print(pad + "|" + " ".join(board[r]) + "|")


if __name__ == "__main__":

    game = PaperSoccer([Human_Player(), AI_Player(Negamax(3))])
    while True:
        # Zamiast game.play() robimy pętlę, żeby móc wyświetlać planszę i komunikaty
        game.show()
        moves = game.possible_moves()
        print("Możliwe ruchy:", moves)
        if game.current_player == 1:
            move = input("Jaki ruch wybierasz? ")
        else:
            move = game.players[1].ask_move(game)
            print(f"AI wybrało ruch: {move}")
        game.make_move(move)
        if game.is_over():
            game.show()
            break
