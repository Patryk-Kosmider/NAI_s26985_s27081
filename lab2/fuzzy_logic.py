import sys
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Zmienne wejściowe
distance = ctrl.Antecedent(np.arange(0, 201, 1), "distance")  # rozszerzone do 200
speed = ctrl.Antecedent(np.arange(-50, 101, 1), "relative_speed")
friction = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "friction")

# Zmienna wyjściowa
braking_force = ctrl.Consequent(np.arange(0, 1.01, 0.01), "braking_force")

# Funkcje przynależności
distance["close"] = fuzz.trimf(distance.universe, [0, 0, 50])
distance["medium"] = fuzz.trimf(distance.universe, [30, 60, 100])
distance["far"] = fuzz.trimf(distance.universe, [80, 200, 200])  # rozszerzone

speed["low"] = fuzz.trimf(speed.universe, [0, 0, 20])
speed["medium"] = fuzz.trimf(speed.universe, [10, 25, 50])
speed["high"] = fuzz.trimf(speed.universe, [30, 60, 100])  # rozszerzone

friction["low"] = fuzz.trimf(friction.universe, [0.0, 0.0, 0.5])
friction["medium"] = fuzz.trimf(friction.universe, [0.25, 0.5, 0.75])
friction["high"] = fuzz.trimf(friction.universe, [0.5, 1.0, 1.0])

braking_force["light"] = fuzz.trimf(braking_force.universe, [0.0, 0.0, 0.5])
braking_force["moderate"] = fuzz.trimf(braking_force.universe, [0.25, 0.5, 0.75])
braking_force["strong"] = fuzz.trimf(braking_force.universe, [0.5, 1.0, 1.0])

# Reguły
rules = [
    ctrl.Rule(distance["close"] & speed["high"], braking_force["strong"]),
    ctrl.Rule(distance["close"] & speed["medium"] & friction["low"], braking_force["strong"]),
    ctrl.Rule(distance["close"] & speed["medium"] & (friction["medium"] | friction["high"]), braking_force["strong"]),
    ctrl.Rule(distance["close"] & friction["low"], braking_force["strong"]),
    ctrl.Rule(distance["close"] & friction["high"], braking_force["moderate"]),
    ctrl.Rule(distance["medium"] & speed["high"], braking_force["moderate"]),
    ctrl.Rule(distance["medium"] & speed["low"], braking_force["light"]),
    ctrl.Rule(distance["medium"] & friction["low"], braking_force["moderate"]),
    ctrl.Rule(distance["far"] & speed["low"], braking_force["light"]),
    ctrl.Rule(distance["far"] & speed["medium"], braking_force["light"]),
    ctrl.Rule(distance["far"] & speed["high"], braking_force["moderate"]),
    ctrl.Rule(distance["close"] & speed["high"] & friction["high"], braking_force["strong"]),
]

# System sterowania
braking_control_system = ctrl.ControlSystem(rules)
braking_simulation = ctrl.ControlSystemSimulation(braking_control_system)

#  Funkcja obliczania hamowania
def calculate_braking_force(dist, rel_speed, fric):

    """
    Oblicza siłę hamowania na podstawie podanych parametrów wejściowych.
    :param dist: funkcja przynależności dla odległości
    :param rel_speed: funkcja przynależności dla predkości względnej
    :param fric: funkcja przynależności dla współczynnika tarcia
    :return: wartość siły hamowania (braking_force)
    """

    dist = max(0, min(200, dist))
    rel_speed = max(-50, min(100, rel_speed))
    fric = max(0.0, min(1.0, fric))

    try:
        braking_simulation.input["distance"] = dist
        braking_simulation.input["relative_speed"] = rel_speed
        braking_simulation.input["friction"] = fric
        braking_simulation.compute()
        val = braking_simulation.output.get("braking_force", 0.0)
    except:
        val = 0.0


    return max(0.0, min(1.0, val))


def membership_functions(dist, rel_speed, fric):
    """
    Rysuje wykresy funkcji przynależności dla wszystkich zmiennych wejściowych i wyjściowych.
    :param dist: funkcja przynależności dla odległości
    :param rel_speed: funkcja przynależności dla predkości względnej
    :param fric: funkcja przynależności dla współczynnika tarcia
    :return: None
    """

    calculate_braking_force(dist, rel_speed, fric)

    plt.ion()
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Dystans")
    for label in distance.terms:
        plt.plot(distance.universe, distance[label].mf, label=label)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Prędkość względna")
    for label in speed.terms:
        plt.plot(speed.universe, speed[label].mf, label=label)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Tarcie")
    for label in friction.terms:
        plt.plot(friction.universe, friction[label].mf, label=label)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Siła hamowania")
    for label in braking_force.terms:
        plt.plot(braking_force.universe, braking_force[label].mf, label=label)
    plt.legend()

    plt.tight_layout()
    plt.show()
    braking_force.view(sim=braking_simulation)
    plt.show()

if __name__ == "__main__":
    sys.exit(0)
