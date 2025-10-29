"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

Symulator asystenta hamowania

Symulator przedstawia działanie systemu asystenta hamowania, który dynamicznie dobiera siłę hamowania w zależności od sytuacji na drodze.
System przewiduje, z jaką siłą należy zahamować, aby uniknąć zderzenia z pojazdem lub przeszkodą znajdującą się przed autem. Decyzja podejmowana jest na podstawie trzech kluczowych danych wejściowych:
Prędkość własnego pojazdu (m/s) – im większa prędkość, tym większe zagrożenie kolizją w przypadku zbyt małego dystansu.
Odległość od przeszkody (m) – określa, jak blisko pojazdu znajduje się obiekt przed nami.
Nawierzchnia (tarcie) – współczynnik przyczepności drogi, wpływający na skuteczność hamowania; np. mokra nawierzchnia wymaga większej siły hamowania, aby zatrzymać pojazd w tym samym czasie.

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install pygame numpy scikit-fuzzy matplotlib

"""

# TODO: umieścić gdzieś rysowanie tych wykresów z fuzzy?

import sys
import pygame
import random
from fuzzy_logic import calculate_braking_force

pygame.init()

# ustawienia pygame
WIDTH, HEIGHT = 1200, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fuzzy logic - symulator hamowania")
clock = pygame.time.Clock()
CAR_W, CAR_H = 70, 40
GROUND_Y = HEIGHT // 2 + 60
PIXELS_PER_M = 5.0
MAX_SPEED = 50.0
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (40, 120, 220)
RED = (200, 30, 30)
font = pygame.font.SysFont("arial", 18)

# zmienne stany
distance_m = 30.0
my_speed = 20.0
lead_speed = 25.0
friction_val = 0.6
brake_force = 0.0
assistant_active = False

my_x = 100
lead_x = my_x + (distance_m + 5) * PIXELS_PER_M + CAR_W

frame_counter = 0
target_speed = lead_speed
state = "setup"  # setup / sim

def road_color_from_friction(f):
    """
    Zwraca kolor nawierzchni zależny od tarcia (ciemniejsza = mniejsze tarcie)
    :param f: współczynnik tarcia (0-1)
    :return: tuple(r, g, b)
    """
    t = max(0.0, min(1.0, f))
    v = int(50 + (160 * t))
    return v, v, v


def braking_deceleration(brake_force, friction):
    """
    Przyszpieszenie hamowania w m/s^2
    :param brake_force: siła hamowania (0-1)
    :param friction: współczynnik tarcia (0-1)
    :return: przyszpieszenie w m/s^2
    """
    g = 9.81
    return brake_force * friction * g


def draw_car(x, color, brake_force=0.0):
    """
    Rysowanie samochodu na ekranie
    :param x: pozycja x auta
    :param color: kolor auta
    :param brake_force: siła hamowania (jako pasek nad autem)
    :return: None
    """
    pygame.draw.rect(screen, color, (int(x), GROUND_Y - CAR_H, CAR_W, CAR_H))
    pygame.draw.rect(screen, BLACK, (int(x), GROUND_Y - CAR_H, CAR_W, CAR_H), 2)
    if brake_force > 0:
        bar_width = int(CAR_W * brake_force)
        bar_color = (255 * brake_force, 255 * (1 - brake_force), 0)
        pygame.draw.rect(
            screen, bar_color, (int(x), GROUND_Y - CAR_H - 10, bar_width, 5)
        )


def draw_hud():
    """
    Rysuje ekran HUD z informacjami o stanie symulacji
    :return: None
    """
    lines = [
        f"Dystans: {distance_m:.1f} m",
        f"Prędkość naszego auta: {my_speed:.1f} m/s",
        f"Prędkość auta przed nami: {lead_speed:.1f} m/s",
        f"Tarcie: {friction_val:.2f}",
        f"Asystent hamowania: {'ON' if assistant_active else 'OFF'}",
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, BLACK)
        screen.blit(txt, (10, 10 + i * 20))


def reset_sim():
    """
    Resetuje stany symulacji
    :return: None
    """
    global my_speed, lead_speed, distance_m, friction_val
    global my_x, lead_x, assistant_active, brake_force, frame_counter, target_speed, state
    my_speed = 20.0
    lead_speed = 25.0
    distance_m = 30.0
    friction_val = 0.6
    my_x = 100
    lead_x = my_x + (distance_m + 5) * PIXELS_PER_M + CAR_W
    assistant_active = False
    brake_force = 0.0
    frame_counter = 0
    target_speed = lead_speed
    state = "setup"


# Pętla główna
running = True
while running:
    dt = clock.tick(60) / 1000.0
    frame_counter += 1

    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                reset_sim()

            # Tryb konfiguracji symulacji
            if state == "setup":
                if event.key == pygame.K_RETURN:
                    state = "sim"
                elif event.key == pygame.K_LEFT:
                    distance_m = max(5.0, distance_m - 1.0)
                elif event.key == pygame.K_RIGHT:
                    distance_m += 1.0
                elif event.key == pygame.K_w:
                    my_speed = min(MAX_SPEED, my_speed + 1.0)
                elif event.key == pygame.K_s:
                    my_speed = max(0.0, my_speed - 1.0)
                elif event.key == pygame.K_UP:
                    lead_speed = min(MAX_SPEED, lead_speed + 1.0)
                elif event.key == pygame.K_DOWN:
                    lead_speed = max(0.0, lead_speed - 1.0)
                elif event.key == pygame.K_a:
                    friction_val = max(0.0, round(friction_val - 0.05, 2))
                elif event.key == pygame.K_d:
                    friction_val = min(1.0, round(friction_val + 0.05, 2))

            # Sterowanie w trakcie symulacji
            elif state == "sim":
                if event.key == pygame.K_w:
                    my_speed = min(MAX_SPEED, my_speed + 2.0)
                elif event.key == pygame.K_s:
                    my_speed = max(0.0, my_speed - 2.0)

    # Symulacja - rysowanie stanu początkowego
    if state == "setup":
        screen.fill(WHITE)
        pygame.draw.rect(
            screen,
            road_color_from_friction(friction_val),
            (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y),
        )
        draw_car(my_x, BLUE)
        draw_car(lead_x, RED)
        draw_hud()
        inst_lines = [
            "Tryb konfiguracji:",
            "←/→ – ustaw dystans",
            "W/S – początkowa prędkość naszego auta",
            "↑/↓ – początkowa prędkość auta przed nami",
            "A/D – tarcie nawierzchni",
            "ENTER – Start symulacji | ESC – Wyjście",
            "M – wyświetl wykresy fuzzy logic",
        ]
        for i, line in enumerate(inst_lines):
            txt = font.render(line, True, BLACK)
            screen.blit(txt, (40, HEIGHT - 110 + i * 20))
        pygame.display.flip()
        continue

    # Sterowanie czerwonym autem (losowe, płynne zmiany prędkości)
    if frame_counter % 60 == 0:
        # Co 60 klatek losujemy nową docelową predkość dla samochodu przed nami
        target_speed = random.uniform(0, MAX_SPEED)
    # Maksymalne przyśpieszenie i opóźnienie samochodu przed nami = 5 m/s, 8 m/s, ograniczamy by nie hamowało 50->0
    MAX_LEAD_ACC = 5.0
    MAX_LEAD_DEC = 8.0
    # Jeśli aktualna prędkość jest mniejsza od docelowej -> przyśpieszamy
    if lead_speed < target_speed:
        lead_speed += MAX_LEAD_ACC * dt
        lead_speed = min(lead_speed, target_speed)
    # Jeśli aktualna prędkość jest większa od docelowej -> hamujemy
    elif lead_speed > target_speed:
        lead_speed -= MAX_LEAD_DEC * dt
        lead_speed = max(lead_speed, target_speed)
    # Aktualizacja pozycji auta przed nami
    # lead_x = przesunięcie w metrach
    lead_x += lead_speed * dt * PIXELS_PER_M

    # Obliczenie dystansu
    # pozycja auta przed nami - (pozycja przodu naszego auta) / pixele (przeliczamy na metry)
    distance_m = max(0.0, (lead_x - (my_x + CAR_W)) / PIXELS_PER_M)

    # Uruchomienie asystenta hamowania
    rel_speed = my_speed - lead_speed
    brake_force = 0.0
    if distance_m < 30 or rel_speed > 0:
        brake_force = calculate_braking_force(distance_m, rel_speed, friction_val)
        # Przyspieszenie hamowania = wynik fuzzy logic * wpsolczynnik tarcia
        decel = braking_deceleration(brake_force, friction_val)
        decel = min(decel, 10.0)
        my_speed -= decel * dt
        my_speed = max(0.0, my_speed)
        assistant_active = True
    else:
        if distance_m > 35:
            my_speed += 5.0 * dt
            my_speed = min(MAX_SPEED, my_speed)
        assistant_active = False
    # Przyrost prędkości auta = prędkość * czas * skala
    my_x += my_speed * dt * PIXELS_PER_M

    # Sprawdzenie kolizji
    if distance_m <= 0:
        # Jeśli dystans do auta na przedzie jest <= 0 - ustawiamy predkość na 0 i pełne hamowanie
        my_speed = 0
        lead_speed = 0
        brake_force = 1.0
        print("Kolizja!")

    # Rysowanie sceny
    offset_x = max(my_x - 150, 0)
    screen.fill(WHITE)
    pygame.draw.rect(
        screen,
        road_color_from_friction(friction_val),
        (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y),
    )
    draw_car(lead_x - offset_x, RED)
    draw_car(my_x - offset_x, BLUE, brake_force)
    draw_hud()

    pygame.display.flip()

pygame.quit()
sys.exit(0)
