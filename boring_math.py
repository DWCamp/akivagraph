import math


def dist(a, b):
    """
    Computes the Euclidean distance between two points
    :param a: The (x, y) coordinates of point a
    :param b: The (x, y) coordinates of point b
    :return: The Euclidean distance between them
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def compute_period(omega_a: float, omega_b: float, world_omega: float):
    """
    Computes an upper bound for the time it will take for the graph to complete one period
    :param omega_a: The rotational speed of rotor A
    :param omega_b: The rotational speed of rotor B
    :param world_omega: The rotational speed of the world
    :return: The time after which the system will have completed a full cycle
    """
    flp = 0.0000001
    flp2 = 2 * flp
    # We only care about absolute values here
    omega_a = abs(omega_a)
    omega_b = abs(omega_b)
    omega_w = abs(world_omega)

    # Ignore zeros by setting equal to the highest
    omega_a = omega_a if omega_a > 0 else max(omega_b, omega_w)
    omega_b = omega_b if omega_b > 0 else max(omega_b, omega_w)
    omega_w = omega_w if omega_w > 0 else max(omega_a, omega_b)

    # Brute force tactic
    slowest = min(omega_a, omega_b, omega_w)
    slow_freq = 360 / slowest
    for i in range(1, 1000000):
        t = i * slow_freq
        if (omega_a * t + flp) % 360 < flp2 and (omega_b * t + flp) % 360 < flp2 and (omega_w * t + flp) % 360 < flp2:
            return t
    print("-------------------------------------------------------------------------\n"
          "ERROR: Could not determine period\n"
          "The omega values used resulted in a period which was too large to compute.\n"
          "\n-------------------------------------------------------------------------")
    exit(-1)
