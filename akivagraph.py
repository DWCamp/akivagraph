"""
Defines the classes needed for the spirograph
"""
import math
import matplotlib.pyplot as plt
# from multiprocessing import Process
import os
import time
from typing import Optional, List, Tuple

from boring_math import dist, compute_period
from config import CONFIG


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


class PointCompute:
    def __init__(self, rotor_a: Rotor, rotor_b: Rotor, world: Rotor,
                 time_step: float, max_steps: int, smoothing: Optional[float], segments: int):
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
        :param segments: Temporary variable while testing multiprocessing implementation
        :return: A list of (x,y) tuples representing all points the pen visited
        """

        self.step = time_step
        self.x = list()
        self.y = list()

        # Compute graph period and calculate time_step / point count
        period = compute_period(rotor_a.omega, rotor_b.omega, world.omega)
        if period / time_step > max_steps:
            self.step = period / max_steps
            report(f"ALERT: Total points needed to calculate the graph at provided time_step ({period / time_step}) "
                   f"exceeds step limit of {max_steps}. Time step has been increased to {round(self.step, 5)}.")

        if CONFIG["DEBUG_LEVEL"] > 0:
            print(f"Period: {period}")

        # Init max / min values
        self.x_max = -1 * math.inf
        self.x_min = math.inf
        self.y_max = -1 * math.inf
        self.y_min = math.inf

        # Generate points
        print("Generating points...")
        segment_len = period / segments
        for i in range(segments):
            start = segment_len * i
            end = start + segment_len
            new_x, new_y = self.compute_segment(start, end, time_step, rotor_a, rotor_b, world, smoothing)
            self.x += new_x
            self.y += new_y

    def __len__(self):
        return len(self.x)

    def compute_segment(self, start: float, end: float, time_step: float,
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
            overlapping = abs(anchor_a[0] - anchor_b[0]) < CONFIG["fl_precision"] \
                and abs(anchor_a[1] - anchor_b[1]) < CONFIG["fl_precision"]
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
                    # Update max and min
                    self.x_max = max(self.x_max, max(x_inter_points))
                    self.x_min = min(self.x_min, min(x_inter_points))
                    self.y_max = max(self.y_max, max(y_inter_points))
                    self.y_min = min(self.y_min, min(y_inter_points))

            # Update prev points
            prev_x, prev_y = x, y

            # Update max and min
            self.x_max = max(self.x_max, x)
            self.x_min = min(self.x_min, x)
            self.y_max = max(self.y_max, y)
            self.y_min = min(self.y_min, y)

            # Add points to list
            x_list.append(x)
            y_list.append(y)

            # Update no_smoothing
            no_smoothing = overlapping

        return x_list, y_list

    def get_tuples(self) -> List[Tuple[float, float]]:
        """
        Returns the x and y data points as a list of tuples
        """
        tuple_list = []
        for x, y in zip(self.x, self.y):
            tuple_list.append((x, y))
        return tuple_list


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
        print(f"Timestamp: {time}")
        print(f"Anchor A: {anchor_a}")
        print(f"Anchor B: {anchor_b}")

    # The translation that all points must be offset by to put anchor A at (0,0)
    trans_offset = (anchor_a[0] * -1, anchor_a[1] * -1)

    if CONFIG["DEBUG_LEVEL"] > 2:
        print(f"T_offset: {trans_offset}")

    # Compute the rotational offset to put anchor b on the x-axis
    anchor_sep = dist(anchor_a, anchor_b)   # Distance between anchors
    trans_anchor_b = (anchor_b[0] + trans_offset[0], anchor_b[1] + trans_offset[1])   # Perform trans offset on anchor b
    delta = dist(trans_anchor_b, (anchor_sep, 0))   # Compute distance moved in rotation
    rot_offset = math.acos((2*(anchor_sep**2) - (delta**2))/(2 * (anchor_sep**2)))  # Find rotational offset
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

    Cx = (AB**2 + AC**2 - BC**2)/(2 * AB)
    cy_component = AC**2 - Cx**2

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
        print(f"INTERPOLATE - {time} | {time_step} | {prev_point} | {curr_point} | {spread}")

    # Compute the midpoint
    mid_time = timestamp - (time_step / 2)
    midpoint = compute_draw_loc(rotor_a, rotor_b, world, mid_time)

    # Recursively compute the other midpoints
    prev_midpoints = interpolate(max_dist, mid_time, time_step / 2, prev_point, midpoint, rotor_a, rotor_b, world)
    next_midpoints = interpolate(max_dist, timestamp, time_step / 2, midpoint, curr_point, rotor_a, rotor_b, world)

    if CONFIG["DEBUG_LEVEL"] > 1:
        print(f"------- RETURN {len(prev_midpoints)} + 1 + {len(next_midpoints)} MID POINTS")
    return [*prev_midpoints, midpoint, *next_midpoints]


def report(msg: str, lpad: str = " ", abort: bool = False) -> None:
    """
    Prints a formatted warning to the console
    :param msg: The alert to print
    :param lpad: A string to pad each line with on the left (Default: ' ')
    :param abort: Exit the program after printing the message
    """
    # Split long messages into 80-char chunks, split along spaces
    lines = []
    words = msg.strip().split(" ")
    curr_line = lpad + words[0]
    for word in words[1:]:
        if len(f"{curr_line} {word}") > 80:
            lines.append(curr_line)
            curr_line = lpad + word
        else:
            curr_line = f"{curr_line} {word}"
    if len(curr_line) > 0:
        lines.append(curr_line)

    # Print message
    print("\n================================================================================")
    for line in lines:
        print(line)
    print("================================================================================\n")
    # Abort if asked
    if abort:
        exit(-1)


def main(radius_a: float, theta_a: float, omega_a: float, length_a: float,
         radius_b: float, theta_b: float, omega_b: float, length_b: float,
         rotor_sep: float, world_x_coord: float, world_y_coord: float, world_omega: float):

    # Parameter validation
    if rotor_sep < 0:
        report("ERROR: Rotor separation must be a positive number", abort=True)
    if rotor_sep + radius_a + radius_a > length_a + length_b:
        report("WARNING: Total linkage length is less than the maximum distance between anchor points. Under many "
               "parameters, this will cause a crash")
    if rotor_sep - (radius_a + radius_b) < abs(length_a - length_b):
        report("WARNING: The difference in linkage length is greater than the minimum distance between anchor points. "
               "Under many parameters, this will cause a crash")

    # Create rotors
    rotor_a = Rotor("A", 0, 0, radius_a, theta_a, omega_a, length_a)
    rotor_b = Rotor("B", rotor_sep, 0, radius_b, theta_b, omega_b, length_b)
    world = Rotor("World", world_x_coord, world_y_coord, 0, 0, world_omega, 0)

    if CONFIG["DEBUG_LEVEL"] > 0:
        print(rotor_a)
        print(rotor_b)
        print(world)

    # Generate points
    start_t = round(time.perf_counter(), 5)
    smoothing = CONFIG["max_point_sep"] if CONFIG["smoothing"] else None
    result = PointCompute(rotor_a, rotor_b, world, CONFIG["step_length"], CONFIG["max_steps"], smoothing, 10)
    points_t = round(time.perf_counter(), 5)
    if CONFIG["print_points"]:
        for x, y in result.get_tuples():
            print(f"{x}\t{y}")
    print(f"{len(result)} POINTS GENERATED IN {points_t - start_t} SECONDS")

    # Compute optimal window bounds
    max_axis_width = max(result.y_max - result.y_min, result.x_max - result.x_min)
    x_center = (result.x_max + result.x_min) / 2
    x_range = ((x_center - max_axis_width / 2) - max_axis_width * CONFIG["padding"],
               (x_center + max_axis_width / 2) + max_axis_width * CONFIG["padding"])
    y_center = (result.y_max + result.y_min) / 2
    y_range = ((y_center - max_axis_width / 2) - max_axis_width * CONFIG["padding"],
               (y_center + max_axis_width / 2) + max_axis_width * CONFIG["padding"])

    # Graph points
    dpi = CONFIG["resolution"] / 10
    plt.figure(dpi=dpi, figsize=(10, 10))  # Set plot dimensions
    ax = plt.subplot(111, aspect='equal')   # Create plot
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)   # Center plot
    ax.plot(result.x, result.y, linewidth=CONFIG["line_width"], color=CONFIG["color"])  # Plot data
    ax.set(xlim=x_range, ylim=y_range)  # Set the range of the graph axes
    plt.axis("off")     # Hide graph axes
    if CONFIG["show_params"]:   # Show rotor parameters
        # The `max_axis_width * 0.01` provides a consistent offset from the image edge
        plt.text(x_range[0] + max_axis_width * CONFIG["param_padding"],
                 y_range[0] + max_axis_width * CONFIG["param_padding"],
                 f"Rotor A: (r: {rotor_a.radius}, θ: {rotor_a.theta}°, ω: {round(rotor_a.omega, 4)}°/s, "
                 f"len: {rotor_a.arm_length})  |  Rotor B: (r: {rotor_b.radius}, θ: {rotor_b.theta}°, "
                 f"ω: {round(rotor_b.omega, 4)}°/s, len: {rotor_b.arm_length})  |  "
                 f"Rotor Sep: {rotor_b.x}  |  World: (x: {world.x}, y: {world.y}, ω: {world.omega}°/s)  |  "
                 f"tΔ: {round(result.step, 3)}  |  "
                 f"smoothing: {CONFIG['max_point_sep'] if CONFIG['smoothing'] else 'Off'}  |  "
                 f"Points: {len(result)}",
                 fontsize="x-small",
                 color=CONFIG["param_color"])

    # Save to output file
    output_file = f"output" + os.path.sep + CONFIG["output_file"]
    plt.savefig(output_file, facecolor=CONFIG["bg_color"])

    # Print plot info to console
    plot_t = round(time.perf_counter(), 5)
    print(f"PLOT RENDERED IN {plot_t - points_t} SECONDS")
    print(f"(TOTAL: {plot_t - start_t} SECONDS)")
    print(f"Output file: {output_file}")

    if CONFIG["DEBUG_LEVEL"] > 0:
        print("\n----------------------------------")
        print(f"X RANGE: {result.x_min}  |  {result.x_max}")
        print(f"Y RANGE: {result.y_min}  |  {result.y_max}")


if __name__ == '__main__':
    # Config validation
    if CONFIG["step_length"] <= 0:
        report("ERROR: Time increment must be a positive number", abort=True)

    if CONFIG["smoothing"] and CONFIG["max_point_sep"] <= 0:
        report("ERROR: max_point_sep must be greater than 0", abort=True)

    if CONFIG["smoothing"] and CONFIG["max_point_sep"] < 0.05:
        report("WARNING: max_point_sep is very low. This may cause long render times or even crashes")

    if not CONFIG["output_file"].endswith("pdf"):   # Resolution doesn't affect PDFs
        if CONFIG["resolution"] < 1:
            report("ERROR: resolution must be greater than 0", abort=True)

        if CONFIG["resolution"] < 100:
            report("WARNING: plot resolution is very low (<100). This will produce very low quality images")

    # Rotor params
    p = {
        "radius_a": 10,
        "theta_a": 90,
        "omega_a": -10.1,
        "length_a": 20,

        "radius_b": 10,
        "theta_b": 0,
        "omega_b": 10,
        "length_b": 20,

        "rotor_sep": 10,

        "world_x_coord": 5,
        "world_y_coord": 30,
        "world_omega": 0,
    }
    main(**p)
