"""
Defines the Rotor class, which contains all the parameters for a rotor

@author - D. William Campman
@date - 2022.09.12
"""

import math


class Rotor:
    """
    Defines one of the spinning disks and linkages that will connect to draw the image
    """
    def __init__(self, name: str, x: float, y: float, radius: float, theta: float, omega: float, arm_length: float):
        """
        :param name: Identifier for telling two rotors apart
        :param x: The x-coordinate of the circle
        :param y: The y-coordinate of the circle
        :param radius: The radius of the circle
        :param theta: The initial rotation of the circle in degrees
        :param omega: The rotational velocity of the circle in degrees per second
        :param arm_length: The length of the linkage attached to the circle
        """
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius
        self.theta = theta
        self.omega = omega
        self.arm_length = arm_length

    def __str__(self) -> str:
        return f"Rotor `{self.name}` | Loc: ({self.x}, {self.y}) | r: {self.radius} " \
               f"| arm length: {self.arm_length} | theta: {self.theta}° | omega: {self.omega}°/sec"

    def theta_at(self, timestamp):
        """
        Computes the theta of the rotor at a given timestamp
        :param timestamp: The timestamp to compute the theta at
        :return: The theta at the given timestamp (in degrees)
        """
        return (self.theta + (self.omega * timestamp)) % 360

    def anchor_loc(self, timestamp):
        """
        Returns the coordinates of the point where the linkage is attached on the perimeter of the disk
        :param timestamp: The timestamp to compute the anchor location at
        :return: A tuple of two floats representing the (x,y) coordinates
        """
        theta_rads = math.radians(self.theta_at(timestamp))
        return (math.cos(theta_rads) * self.radius + self.x), (math.sin(theta_rads) * self.radius + self.y)
