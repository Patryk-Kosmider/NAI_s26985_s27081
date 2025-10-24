"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

Symulator asystenta hamowania

System przewiduje, z jaką siłą trzeba zahamować, aby uniknąć kolizji z przeszkodą. System przyjmuje trzy dane wejściowe: prędkość, odległość od przeszkody oraz nawierzchnię (tarcie).

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install pygame numpy scikit-fuzzy matplotlib

"""

import sys

import pygame

from fuzzy_logic import calculate_braking_force

pygame.init()

# TODO: naprawic hamowanie - dziala ale blednie pokazuje rodzaj hamowania
# TODO: dokrecic hamowanie - samochod wyrabia przed siciana
# TODO: jakies efekty/modele zeby to wygladalo
# TODO: moze dodac km/h i funkcje konwertujace??? wzialem m/s bo latwiej na wzorach operowac

# ekran
WIDTH, HEIGHT = 800, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fuzzy logic - symulator hamowania")
clock = pygame.time.Clock()

# wielkosc obiektow
CAR_W, CAR_H = 70, 40
WALL_W, WALL_H = 40, 100
GROUND_Y = HEIGHT // 2 + 60

# skala 1 m to 5 pikseli
PIXELS_PER_M = 5.0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (40, 120, 220)
RED = (200, 30, 30)
DARK_RED = (140, 20, 20)

font = pygame.font.SysFont("arial", 18)
large_font = pygame.font.SysFont("arial", 28, bold=True)

# wartosci domyslne
default_distance = 30.0
distance_m = default_distance
speed = 20.0
friction_val = 0.6

# Derived positions
wall_x = WIDTH - 50
wall_y = GROUND_Y - WALL_H


def dist_to_car_x(dist_m):
    """Return car_x (left of car) such that distance from car front to wall = dist_m"""
    car_front_x = wall_x - dist_m * PIXELS_PER_M
    car_x = car_front_x - CAR_W
    return car_x


# wartosci domyslne
state = "setup"
car_x = dist_to_car_x(distance_m)
car_y = GROUND_Y - CAR_H
brake_force = 0.0
brake_category = "none"


# kolor drogi zaleźnie od tarcia - 0.0 czarny, 1.0 jasny szary
def road_color_from_friction(f):
    t = max(0.0, min(1.0, f))
    v = int(40 + (160 * t))
    return v, v, v


# Draw HUD
def draw_hud():
    hud_x = WIDTH // 2
    lines = [
        f"Dystans: {distance_m:.1f} m",
        f"Prędkosc: {speed:.1f} m/s",
        f"Tarcie: {friction_val:.2f}",
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, BLACK)
        rect = txt.get_rect(center=(hud_x, 30 + i * 20))
        screen.blit(txt, rect)

    # Center big: braking value during sim/stopped
    if state in ("sim", "stopped"):
        bf_text = large_font.render(f"Hamowanie: {brake_force:.2f}", True, BLACK)
        bf_rect = bf_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
        screen.blit(bf_text, bf_rect)

    # Instructions
    inst_lines = [
        "Sterowanie:",
        "←/→ Dystans  W/S Prędkosc  A/D Tarcie",
        "ENTER: Start  ESC: Wyjscie"
    ]
    for i, line in enumerate(inst_lines):
        txt = font.render(line, True, BLACK)
        screen.blit(txt, (20, HEIGHT - 70 + i * 18))


def braking_deceleration(brake_force, friction):
    g = 9.81  # m/s^2
    return brake_force * friction * g


running = True
while running:
    dt = clock.tick(60) / 1_000.0  # sekundy
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if state == "setup":
                if event.key == pygame.K_RETURN:
                    # oblicz sile hamowania
                    car_x = dist_to_car_x(distance_m)
                    brake_force = calculate_braking_force(distance_m, speed, friction_val)

                    # kategorie hamowania
                    if brake_force >= 0.7:
                        brake_category = "strong"
                    elif brake_force >= 0.4:
                        brake_category = "moderate"
                    else:
                        brake_category = "light"

                    state = "sim"
                # ustawianie parametrow
                if event.key == pygame.K_RIGHT:
                    distance_m = max(0.0, distance_m + 1.0)
                if event.key == pygame.K_LEFT:
                    distance_m = max(0.0, distance_m - 1.0)
                if event.key == pygame.K_w:
                    speed = min(40.0, speed + 1.0)
                if event.key == pygame.K_s:
                    speed = max(0.0, speed - 1.0)
                if event.key == pygame.K_d:
                    friction_val = min(1.0, round(friction_val + 0.02, 2))
                if event.key == pygame.K_a:
                    friction_val = max(0.0, round(friction_val - 0.02, 2))
    # tlo
    screen.fill(WHITE)

    # droga
    road_color = road_color_from_friction(friction_val)
    pygame.draw.rect(screen, road_color, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))

    # sciana
    pygame.draw.rect(screen, RED, (wall_x, wall_y, WALL_W, WALL_H))
    pygame.draw.rect(screen, DARK_RED, (wall_x, wall_y + WALL_H - 8, WALL_W, 8))

    if state == "setup":
        car_x = dist_to_car_x(distance_m)
        pygame.draw.rect(screen, BLUE, (car_x, car_y, CAR_W, CAR_H))

    elif state == "sim":
        # zwalnianie m/s^2
        decel = braking_deceleration(brake_force, friction_val)

        # akutalizowanie hamowania (v = v - a * dt)
        speed -= decel * dt
        speed = max(speed, 0.0)

        # move car
        car_x += speed * dt * PIXELS_PER_M

        pygame.draw.rect(screen, BLUE, (int(car_x), int(car_y), CAR_W, CAR_H))

        if speed <= 0.0:
            state = "stopped"

    elif state == "stopped":
        pygame.draw.rect(screen, BLUE, (int(car_x), int(car_y), CAR_W, CAR_H))

        if brake_category == "strong":
            st = large_font.render("Silne hamowanie", True, DARK_RED)
            screen.blit(st, (WIDTH // 2 - st.get_width() // 2, HEIGHT // 2 + 10))
        elif brake_category == "moderate":
            st = large_font.render("Srednie hamowanie", True, (80, 80, 80))
            screen.blit(st, (WIDTH // 2 - st.get_width() // 2, HEIGHT // 2 + 10))
        else:
            st = large_font.render("Lekkie hamowanie", True, (40, 100, 40))
            screen.blit(st, (WIDTH // 2 - st.get_width() // 2, HEIGHT // 2 + 10))

    draw_hud()
    pygame.display.flip()

pygame.quit()
sys.exit(0)
