import sys

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# TODO: Wstęp z autorami / opisem projektu

# zmienne wejsciowe
distance = ctrl.Antecedent(np.arange(0, 151, 1), "distance")
speed = ctrl.Antecedent(np.arange(-50, 101, 1), "relative_speed")
friction = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "friction")

# zmienna wyjsciowe
braking_force = ctrl.Consequent(np.arange(0, 1.01, 0.01), "braking_force")

# funkcje przynaleznosci
distance["close"] = fuzz.trimf(distance.universe, [0, 0, 75])
distance["medium"] = fuzz.trimf(distance.universe, [25, 75, 125])
distance["far"] = fuzz.trimf(distance.universe, [75, 150, 150])

speed["low"] = fuzz.trimf(speed.universe, [0, 0, 20])
speed["medium"] = fuzz.trimf(speed.universe, [0, 20, 40])
speed["high"] = fuzz.trimf(speed.universe, [20, 40, 40])

friction["low"] = fuzz.trimf(friction.universe, [0.0, 0.0, 0.5])
friction["medium"] = fuzz.trimf(friction.universe, [0.25, 0.5, 0.75])
friction["high"] = fuzz.trimf(friction.universe, [0.5, 1.0, 1.0])

braking_force["light"] = fuzz.trimf(braking_force.universe, [0.0, 0.0, 0.5])
braking_force["moderate"] = fuzz.trimf(braking_force.universe, [0.25, 0.5, 0.75])
braking_force["strong"] = fuzz.trimf(braking_force.universe, [0.5, 1.0, 1.0])

# reguly
rule1 = ctrl.Rule(
    distance["close"] & speed["high"] & friction["low"],
    braking_force["strong"],
)
rule2 = ctrl.Rule(
    distance["close"] & speed["medium"] & friction["medium"],
    braking_force["moderate"],
)
rule3 = ctrl.Rule(
    distance["medium"] & speed["medium"] & friction["high"],
    braking_force["light"],
)
rule4 = ctrl.Rule(distance["far"] | speed["low"], braking_force["light"])

# system sterowania
braking_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
braking_simulation = ctrl.ControlSystemSimulation(braking_control_system)


def calculate_braking_force(dist, rel_speed, fric):
    """
    Oblicza siłę hamowania na podstawie podanych parametrów wejściowych.
    :param dist: funkcja przynależności dla odległości
    :param rel_speed: funkcja przynależności dla predkości względnej
    :param fric: funkcja przynależności dla współczynnika tarcia
    :return: wartość siły hamowania (braking_force)
    """
    braking_simulation.input["distance"] = dist
    braking_simulation.input["relative_speed"] = rel_speed
    braking_simulation.input["friction"] = fric

    braking_simulation.compute()
    return braking_simulation.output["braking_force"]


def membership_functions():
    """
    Rysuje wykresy funkcji przynależności dla wszystkich zmiennych wejściowych i wyjściowych.
    :return: None
    """
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Dystans - funkcja członkostwa")
    for label in distance.terms:
        plt.plot(distance.universe, distance[label].mf, label=label)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Prędkość względna - funkcja członkostwa")
    for label in speed.terms:
        plt.plot(speed.universe, speed[label].mf, label=label)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Prędkość względna - funkcja członkostwa")
    for label in friction.terms:
        plt.plot(friction.universe, friction[label].mf, label=label)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Siła hamowania - funkcja członkostwa")
    for label in braking_force.terms:
        plt.plot(braking_force.universe, braking_force[label].mf, label=label)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # wykres wyniku
    braking_force.view(sim=braking_simulation)
    plt.show()


if __name__ == "__main__":
    sys.exit(0)
