"""
Specifies the PointCompute class and its return type

@author - D. William Campman
@date - 2022.09.12
"""

import math
from multiprocessing import Pool
import os
from typing import Optional, List, Tuple

from rotor import Rotor
from utils import dist, fl_eq, report
from config import CONFIG


class PointCompute:

    def __init__(self, rotor_a: Rotor, rotor_b: Rotor, world: Rotor,
                 time_step: float, max_steps: int, smoothing: Optional[float]):
        """
        Generates a list of points for the scatter plot
        :param rotor_a: The main rotor
        :param rotor_b: The secondary rotor
        :param world: The world underneath the rotors
        :param time_step: The amount of time between point calculations
        :param max_steps: The limit on the number of steps. If the number of steps
        required to draw the plot exceeds this, time_step will be increased as needed
        :param smoothing: The distance above which adjacent points should be interpolated.
        To disable smoothing, pass `None`
        :return: A list of (x,y) tuples representing all points the pen visited
        """

        self.step = time_step
        self.x = list()
        self.y = list()

        # Compute graph period

        period = None

        # We only care about absolute values here
        omega_a = abs(rotor_a.omega)
        omega_b = abs(rotor_b.omega)
        omega_w = abs(world.omega)

        # Remove zeros by setting any zero values equal to the highest other value
        omega_a = omega_a if omega_a > 0 else max(omega_b, omega_w)
        omega_b = omega_b if omega_b > 0 else max(omega_b, omega_w)
        omega_w = omega_w if omega_w > 0 else max(omega_a, omega_b)

        # Brute force tactic
        slowest = min(omega_a, omega_b, omega_w)
        slow_freq = 360 / slowest
        for i in range(1, 1000000):
            t = i * slow_freq
            # If all omegas
            if fl_eq(omega_a * t, 0, mod=360) and fl_eq(omega_b * t, 0, mod=360) and fl_eq(omega_w * t, 0, mod=360):
                period = t
                break

        if CONFIG["DEBUG_LEVEL"] > 0:
            print(f"Period: {period}")

        if period is None:
            report("ERROR: Could not determine period. The omega values used resulted in a period "
                   "which was too large to compute.", abort=True)

        # Calculate time_step & point count
        req_points = period / time_step
        if req_points > max_steps:
            time_step = period / max_steps
            report(f"ALERT: At provided time_step, the total points needed to calculate the graph ({req_points}) "
                   f"exceeds step limit of {max_steps}. Time step has been increased to {round(time_step, 5)}.")

        # Init max / min values
        self.x_max = -1 * math.inf
        self.x_min = math.inf
        self.y_max = -1 * math.inf
        self.y_min = math.inf

        # Compute task count
        if CONFIG["disable_threading"]:
            tasks = 1
        elif smoothing is None:
            tasks = 1 if max_steps <= 11000 else os.cpu_count()
        else:
            tasks = 1 if max_steps <= 7500 else os.cpu_count() * 10

        if CONFIG["DEBUG_LEVEL"] > 0:
            print(f"Tasks: {tasks}")

        # Generate points

        if tasks <= 1:  # For only 1 task, just do it single threaded
            self.x, self.y = compute_segment(0, period, time_step, rotor_a, rotor_b, world, smoothing)

            # Update max/min
            self.x_max = max(self.x)
            self.x_min = min(self.x)
            self.y_max = max(self.y)
            self.y_min = min(self.y)
        else:
            segment_len = period / tasks
            seg_params = list()
            for i in range(tasks):
                start = segment_len * i
                end = start + segment_len
                seg_params.append((start, end, time_step, rotor_a, rotor_b, world, smoothing))

            with Pool() as pool:
                result = pool.starmap(compute_segment, seg_params)

            for new_x, new_y in result:
                self.x += new_x
                self.y += new_y

                # Update max/min
                self.x_max = max(self.x_max, max(new_x))
                self.x_min = min(self.x_min, min(new_x))
                self.y_max = max(self.y_max, max(new_y))
                self.y_min = min(self.y_min, min(new_y))

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.x):
            self.n += 1
            return self.x[self.n], self.y[self.n]


def compute_segment(start: float, end: float, time_step: float,
                    rotor_a: Rotor, rotor_b: Rotor, world: Rotor,
                    smoothing: Optional[float]) -> Tuple[List[float], List[float]]:
    """
    Renders a segment of the graph between two timestamps (inclusive) at a given resolution
    :param start: The timestamp of the first point to graph
    :param end: The timestamp of the last point to graph
    :param time_step: The time resolution to graph at
    :param rotor_a: The primary rotor
    :param rotor_b: The secondary rotor
    :param world: The world object
    :param smoothing: The threshold for point interpolation. Pass `None` for no smoothing
    :return: The list of x and y points
    """

    # Init loop tracking values
    curr_time = start
    no_smoothing = False

    # Init point store values
    x, y = compute_draw_loc(rotor_a, rotor_b, world, curr_time)
    prev_x, prev_y = x, y
    x_list = [x]
    y_list = [y]

    # Compute points
    while curr_time < end:
        # Increment time
        curr_time += time_step

        # Compute next point
        anchor_a = rotor_a.anchor_loc(curr_time)
        anchor_b = rotor_b.anchor_loc(curr_time)
        overlapping = fl_eq(anchor_a[0], anchor_b[0]) and fl_eq(anchor_a[1], anchor_b[1])
        no_smoothing = True if overlapping else no_smoothing
        # If anchor points are on top of each other, this can cause indeterminate results
        # In this case, just reuse the previous point
        if overlapping:
            x = x_list[-1]
            y = y_list[-1]
        else:
            x, y = compute_draw_loc(rotor_a, rotor_b, world, curr_time)

        if smoothing is not None and no_smoothing <= 0:   # Do not interpolate when anchors overlap
            # Interpolate
            midpoints = interpolate(smoothing, curr_time, time_step,
                                    (prev_x, prev_y), (x, y),
                                    rotor_a, rotor_b, world)
            x_inter_points = [point[0] for point in midpoints]
            y_inter_points = [point[1] for point in midpoints]

            if len(x_inter_points) > 0:
                # Add interpolated points to list
                x_list += x_inter_points
                y_list += y_inter_points

        # Update prev points
        prev_x, prev_y = x, y

        # Add points to list
        x_list.append(x)
        y_list.append(y)

        # Update no_smoothing
        no_smoothing = overlapping

    return x_list, y_list


def compute_draw_loc(rotor_a: Rotor, rotor_b: Rotor, world: Rotor, timestamp: float):
    """
    Computes the coordinate of the drawing point where the two rotor linkages connect
    :param rotor_a: The primary rotor
    :param rotor_b: The secondary rotor
    :param world: The world
    :param timestamp: The current timestamp
    :return: The coordinates of the meeting point
    """

    # Get relevant values
    anchor_a = rotor_a.anchor_loc(timestamp)
    anchor_b = rotor_b.anchor_loc(timestamp)

    arm_length_a = rotor_a.arm_length
    arm_length_b = rotor_b.arm_length

    if CONFIG["DEBUG_LEVEL"] > 2:
        print("==============================================")
        print(f"Timestamp: {timestamp}")
        print(f"Anchor A: {anchor_a}")
        print(f"Anchor B: {anchor_b}")

    # The translation that all points must be offset by to put anchor A at (0,0)
    trans_offset = (anchor_a[0] * -1, anchor_a[1] * -1)

    if CONFIG["DEBUG_LEVEL"] > 2:
        print(f"T_offset: {trans_offset}")

    # Compute the rotational offset to put anchor b on the x-axis
    anchor_sep = dist(anchor_a, anchor_b)  # Distance between anchors
    trans_anchor_b = (anchor_b[0] + trans_offset[0], anchor_b[1] + trans_offset[1])  # Perform trans offset on anchor b
    delta = dist(trans_anchor_b, (anchor_sep, 0))  # Compute distance moved in rotation
    rot_offset = math.acos((2 * (anchor_sep ** 2) - (delta ** 2)) / (2 * (anchor_sep ** 2)))  # Find rotational offset
    mapped_anchor_b = (delta, 0)

    # Make sure the anchors aren't too close or too far apart for the linkage lengths
    if anchor_sep < abs(arm_length_a - arm_length_b):
        report("ERROR: CONSTRAINT VIOLATION - Anchors are closer together than the difference between their arm "
               "lengths. Try increasing the rotor spread or reducing their radii", abort=True)
    if anchor_sep > arm_length_a + arm_length_b:
        report("ERROR: CONSTRAINT VIOLATION - Anchors are further apart than the sum of their arm lengths. Try "
               "reducing rotor spread or increasing arm length", abort=True)

    if CONFIG["DEBUG_LEVEL"] > 2:
        print(f"R_offset: {rot_offset}")
        print(f"Translated Anchor B: {trans_anchor_b}")
        print(f"Rotated Anchor B: {mapped_anchor_b}")

    # Offset formula always gives a positive result. If anchor b was above the x-axis, make offset negative
    if trans_anchor_b[1] > 0:
        rot_offset *= -1

    # Compute location of drawing point using formula found here -> https://math.stackexchange.com/a/1989113
    AB = anchor_sep
    AC = arm_length_a
    BC = arm_length_b

    Cx = (AB ** 2 + AC ** 2 - BC ** 2) / (2 * AB)
    cy_component = AC ** 2 - Cx ** 2

    if CONFIG["DEBUG_LEVEL"] > 2:
        print(f"cy_component: {cy_component}")

    Cy = math.sqrt(abs(cy_component)) * (-1 if cy_component < 0 else 1)

    if CONFIG["DEBUG_LEVEL"] > 2:
        print(f"Rotor draw point: ({Cx}, {Cy})")

    # Undo coordinate transformations
    Cx_trans = Cx * math.cos(-rot_offset) - Cy * math.sin(-rot_offset)
    Cy_trans = Cx * math.sin(-rot_offset) + Cy * math.cos(-rot_offset)

    Cx_trans -= trans_offset[0]
    Cy_trans -= trans_offset[1]

    if CONFIG["DEBUG_LEVEL"] > 2:
        print(f"Translated draw point: ({Cx_trans}, {Cy_trans})")

    # Do the hokey-pokey and turn that point around
    world_rad = math.radians(world.theta_at(timestamp))
    Cx_world = (Cx_trans - world.x) * math.cos(world_rad) - (Cy_trans - world.y) * math.sin(world_rad)
    Cy_world = (Cx_trans - world.x) * math.sin(world_rad) + (Cy_trans - world.y) * math.cos(world_rad)

    Cx_world += world.x
    Cy_world += world.y

    if CONFIG["DEBUG_LEVEL"] > 2:
        print("==============================================")
        print(f"cy_component: {cy_component}")
        print(f"World draw point: ({Cx_world}, {Cy_world})")

    return Cx_world, Cy_world


def interpolate(max_dist: float, timestamp: float, time_step: float,
                prev_point: Tuple[float, float], curr_point: Tuple[float, float],
                rotor_a: Rotor, rotor_b: Rotor, world: Rotor) -> List[Tuple[float, float]]:
    """
    Recursively generates more and more precise midpoints between two points until
    each one is separated from the next by less than the maximum allowable distance
    :param max_dist: The maximum allowed separation between two points
    :param timestamp: The time at the current point
    :param time_step: The current time step (when called from another function, this is CONFIG["time_step"])
    :param prev_point: The coordinates of the previous point
    :param curr_point: The coordinates of the current point
    :param rotor_a: Rotor A
    :param rotor_b: Rotor B
    :param world: The world
    :return: A list of interstitial points. If points are suitably close, the list will be empty
    """
    # Base condition
    spread = dist(prev_point, curr_point)
    if spread < max_dist:
        return []

    if CONFIG["DEBUG_LEVEL"] > 1:
        print(f"INTERPOLATE - {timestamp} | {time_step} | {prev_point} | {curr_point} | {spread}")

    # Compute the midpoint
    mid_time = timestamp - (time_step / 2)
    midpoint = compute_draw_loc(rotor_a, rotor_b, world, mid_time)

    # Recursively compute the other midpoints
    prev_midpoints = interpolate(max_dist, mid_time, time_step / 2, prev_point, midpoint, rotor_a, rotor_b, world)
    next_midpoints = interpolate(max_dist, timestamp, time_step / 2, midpoint, curr_point, rotor_a, rotor_b, world)

    if CONFIG["DEBUG_LEVEL"] > 1:
        print(f"------- RETURN {len(prev_midpoints)} + 1 + {len(next_midpoints)} MID POINTS")
    return [*prev_midpoints, midpoint, *next_midpoints]
