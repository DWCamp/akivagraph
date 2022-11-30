"""
Defines the classes needed for the spirograph
"""
import math
import matplotlib.pyplot as plt
from datetime import datetime
from boring_math import dist


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
        self.theta_start = theta
        self.omega = omega
        self.arm_length = arm_length

    def __str__(self) -> str:
        return f"Rotor `{self.name}` | Loc: ({self.x}, {self.y}) | r: {self.radius} " \
               f"| arm length: {self.arm_length} | theta: {self.theta}° | omega: {self.omega}°/sec"

    def __eq__(self, other):
        # The use of the inequality on theta is to avoid floating point errors throwing off the comparison
        return other.name == self.name and other.x == self.x and other.y == self.y and other.radius == self.radius \
                and other.arm_length == self.arm_length and other.omega == self.omega \
                and abs(other.theta - self.theta) < CONFIG["fl_precision"]

    def copy(self):
        """
        Returns a duplicate copy of this rotor
        """
        return Rotor(self.name, self.x, self.y, self.radius, self.theta, self.omega, self.arm_length)

    def increment(self, time_step):
        """
        Spins the rotor by a defined time increment
        :param time_step: The amount of time in seconds to spin the rotor. Lower values give more accurate simulations
        """
        self.theta = (self.theta + self.omega * time_step) % 360

    def anchor_loc(self, time=None):
        """
        Returns the coordinates of the point where the linkage is attached on the perimeter of the disk
        :param time: If provided, the anchor coordinates will be computed for where the rotor
        would be at that moment (Default: None)
        :return: A tuple of two floats representing the (x,y) coordinates
        """
        theta_rads = math.radians(self.theta if time is None else (self.theta_start + (self.omega * time)) % 360)
        return (math.cos(theta_rads) * self.radius + self.x), (math.sin(theta_rads) * self.radius + self.y)


class PlotResult:
    def __init__(self, rotor_a: Rotor, rotor_b: Rotor, world: Rotor):
        """
        Generates a list of points for the scatter plot
        :param rotor_a: The main rotor
        :param rotor_b: The secondary rotor
        :param world: The world underneath the rotors
        :return: A list of (x,y) tuples representing all of the points the pen visited
        """
        self.aborted = False

        # store initial parameters so we can check for a cycle
        rotor_a_init = rotor_a.copy()
        rotor_b_init = rotor_b.copy()
        world_init = world.copy()
        time = 0

        # Store initial point before going into loop
        x, y = compute_draw_loc(rotor_a.anchor_loc(), rotor_a.arm_length,
                                rotor_b.anchor_loc(), rotor_b.arm_length,
                                world)
        self.x = [x]
        self.y = [y]

        self.x_max = x
        self.x_min = x
        self.y_max = y
        self.y_min = y

        prev_x, prev_y = x, y

        print("Generating points...")

        # Generate points
        for _ in range(CONFIG["num_points"]):
            # Step both rotors forwards
            rotor_a.increment(CONFIG["time_step"])
            rotor_b.increment(CONFIG["time_step"])
            world.increment(CONFIG["time_step"])
            time += CONFIG["time_step"]

            # Compute next point
            anchor_a = rotor_a.anchor_loc()
            anchor_b = rotor_b.anchor_loc()

            if abs(anchor_a[0] - anchor_b[0]) < CONFIG["fl_precision"] \
                    and abs(anchor_a[1] - anchor_b[1]) < CONFIG["fl_precision"]:
                # If anchor points are on top of each other, this can cause indeterminate results
                # In this case, just reuse the previous point
                x = self.x[-1]
                y = self.y[-1]
            else:
                x, y = compute_draw_loc(rotor_a.anchor_loc(), rotor_a.arm_length,
                                        rotor_b.anchor_loc(), rotor_b.arm_length,
                                        world)

            if CONFIG["smoothing"]:
                pass
            else:
                self.x.append(x)
                self.y.append(y)

                # Update max and min
                self.x_max = max(self.x_max, x)
                self.x_min = min(self.x_min, x)
                self.y_max = max(self.y_max, y)
                self.y_min = min(self.y_min, y)

            # Abort if both rotors and the world have returned to their initial states
            if CONFIG["abort_on_cycle"] and rotor_a == rotor_a_init and rotor_b == rotor_b_init and world == world_init:
                self.aborted = True
                break

    def __len__(self):
        return len(self.x)

    def get_tuples(self) -> [(float, float)]:
        """
        Returns the x and y data points as a list of tuples
        """
        tuple_list = []
        for x, y in zip(self.x, self.y):
            tuple_list.append((x, y))
        return tuple_list


def compute_draw_loc(anchor_a: (float, float), arm_length_a: float,
                     anchor_b: (float, float), arm_length_b: float,
                     world: Rotor):
    """
    Computes the coordinate of the drawing point where the two rotor linkages connect
    :param anchor_a: The (x,y) coordinates of rotor a's anchor point
    :param arm_length_a: The length of rotor a's arm
    :param anchor_b: The (x,y) coordinates of rotor b's anchor point
    :param arm_length_b: The length of rotor b's arm
    :param world: The world which rotates under the other rotors
    :return: The coordinates of the meeting point
    """

    if CONFIG["DEBUG"]:
        print("==============================================")
        print(f"Anchor A: {anchor_a}")
        print(f"Anchor B: {anchor_b}")

    # The translation that all points must be offset by to put anchor a at (0,0)
    trans_offset = (anchor_a[0] * -1, anchor_a[1] * -1)

    if CONFIG["DEBUG"]:
        print(f"T_offset: {trans_offset}")

    # Compute the rotational offset to put anchor b on the x-axis
    anchor_sep = dist(anchor_a, anchor_b)   # Distance between anchors
    trans_anchor_b = (anchor_b[0] + trans_offset[0], anchor_b[1] + trans_offset[1])   # Perform trans offset on anchor b
    delta = dist(trans_anchor_b, (anchor_sep, 0))   # Compute distance moved in rotation
    rot_offset = math.acos((2*(anchor_sep**2) - (delta**2))/(2 * (anchor_sep**2)))  # Find rotational offset
    mapped_anchor_b = (delta, 0)

    # Make sure the anchors aren't too close or too far apart for the linkage lengths
    if anchor_sep < arm_length_a - arm_length_b:
        print("-------------------------------------------------------------------------\n"
              "ERROR: CONSTRAINT VIOLATION\n"
              "Anchors are closer together than the difference between their arm lengths\n"
              "Try increasing the rotor spread or reducing their radii"
              "\n-------------------------------------------------------------------------")
        exit(-1)
    if anchor_sep > arm_length_a + arm_length_b:
        print("-------------------------------------------------------------------------\n"
              "ERROR: CONSTRAINT VIOLATION\n"
              "Anchors are further apart than the sum of their arm lengths\n"
              "Try reducing rotor spread or increasing arm length"
              "\n-------------------------------------------------------------------------")
        exit(-1)

    if CONFIG["DEBUG"]:
        print(f"R_offset: {rot_offset}")

    # Offset formula always gives a positive result. If anchor b was above the x-axis, make offset negative
    if trans_anchor_b[1] > 0:
        rot_offset *= -1

    # Compute location of drawing point using formula found here -> https://math.stackexchange.com/a/1989113
    AB = anchor_sep
    AC = arm_length_a
    BC = arm_length_b

    Cx = (AB**2 + AC**2 - BC**2)/(2 * AB)
    cy_component = AC**2 - Cx**2
    Cy = math.sqrt(abs(cy_component)) * (-1 if cy_component < 0 else 1)

    if CONFIG["DEBUG"]:
        print(f"Rotor draw point: ({Cx}, {Cy})")

    # Undo coordinate transformations
    Cx_trans = Cx * math.cos(-rot_offset) - Cy * math.sin(-rot_offset)
    Cy_trans = Cx * math.sin(-rot_offset) + Cy * math.cos(-rot_offset)

    Cx_trans -= trans_offset[0]
    Cy_trans -= trans_offset[1]

    if CONFIG["DEBUG"]:
        print(f"Translated draw point: ({Cx_trans}, {Cy_trans})")

    # Do the hokey-pokey and turn that point around
    world_rad = math.radians(world.theta)
    Cx_world = (Cx_trans - world.x) * math.cos(world_rad) - (Cy_trans - world.y) * math.sin(world_rad)
    Cy_world = (Cx_trans - world.x) * math.sin(world_rad) + (Cy_trans - world.y) * math.cos(world_rad)

    Cx_world += world.x
    Cy_world += world.y

    if CONFIG["DEBUG"]:
        print(f"World draw point: ({Cx_world}, {Cy_world})")

    if CONFIG["VERBOSE"] and not CONFIG["DEBUG"] and abs(Cy_world) > 100:
        print("==============================================")
        print(f"Anchor A: {anchor_a}")
        print(f"Anchor B: {anchor_b}")
        print(f"Translated Anchor B: {trans_anchor_b}")
        print(f"Rotated Anchor B: {mapped_anchor_b}")
        print(f"T_offset: {trans_offset}")
        print(f"R_offset: {rot_offset}")
        print(f"cy_component: {cy_component}")
        print(f"Rotor draw point: ({Cx}, {Cy})")
        print(f"Translated draw point: ({Cx_trans}, {Cy_trans})")
        print(f"World draw point: ({Cx_world}, {Cy_world})")

    return Cx_world, Cy_world


def interpolate(time: float, time_step: float,
                prev_point: (float, float), curr_point: (float, float),
                rotor_a: Rotor, rotor_b: Rotor) -> [(float, float)]:
    """
    Recursively generates more and more precise midpoints between two points until
    each one is separated from the next by less than the maximum allowable distance
    :param time: The current time
    :param time_step: The current time step (when called from another function, this is CONFIG["time_step"])
    :param prev_point: The coordinates of the previous point
    :param curr_point: The coordinates of the current point
    :param rotor_a: Rotor A
    :param rotor_b: Rotor B
    :return: A list of interstitial points. If points are suitably close, the list will be empty
    """
    point_list = []
    # Base condition
    if dist(prev_point, curr_point) < CONFIG["max_point_sep"]:
        return point_list

    # Compute mid time
    mid_time = time - (time_step / 2)
    mid_anchor_a = rotor_a.anchor_loc(mid_time)
    mid_anchor_b = rotor_b.anchor_loc(mid_time)
    x, y = compute_draw_loc(mid_anchor_a, rotor_a.arm_length, mid_anchor_b, rotor_b.arm_length, world)


def main(radius_a: float,
         theta_a: float,
         omega_a: float,
         length_a: float,
         radius_b: float,
         theta_b: float,
         omega_b: float,
         length_b: float,
         rotor_sep: float,
         world_x_coord: float,
         world_y_coord: float,
         world_omega: float):
    global CONFIG
    # Parameter validation
    if rotor_sep < 0:
        print("-----------------------------------------------------\n"
              "ERROR: Rotor separation must be a positive number"
              "\n-----------------------------------------------------")
        exit(-1)
    if rotor_sep + radius_a + radius_a > length_a + length_b:
        print("-----------------------------------------------------\n"
              "WARNING: Total linkage length is less than the maximum distance between anchor points. "
              "Under many parameters, this will cause a crash"
              "\n-----------------------------------------------------")
    if rotor_sep - (radius_a + radius_b) < abs(length_a - length_b):
        print("-----------------------------------------------------\n"
              "WARNING: The difference in linkage length is greater than the minimum distance between anchor points. "
              "Under many parameters, this will cause a crash"
              "\n-----------------------------------------------------")

    # Create rotors
    rotor_a = Rotor("A", 0, 0, radius_a, theta_a, omega_a, length_a)
    rotor_b = Rotor("B", rotor_sep, 0, radius_b, theta_b, omega_b, length_b)
    world = Rotor("World", world_x_coord, world_y_coord, 0, 0, world_omega, 0)

    if CONFIG["DEBUG"]:
        print(rotor_a)
        print(rotor_b)

    # Generate points
    start = datetime.now()
    result = PlotResult(rotor_a, rotor_b, world)
    points = datetime.now()
    if CONFIG["PRINT_RESULTS"]:
        for x, y in result.get_tuples():
            print(f"{x}\t{y}")
    print(f"ABORTED EARLY AT {len(result.x)} POINTS" if result.aborted
          else f"RAN TO COMPLETION ({CONFIG['num_points']} POINTS)")
    print(f"POINTS GENERATED IN {(points - start).total_seconds()} SECONDS")

    # Compute optimal window bounds
    max_axis_width = max(result.y_max - result.y_min, result.x_max - result.x_min)
    x_center = (result.x_max + result.x_min) / 2
    x_range = ((x_center - max_axis_width / 2) - max_axis_width * CONFIG["padding"],
               (x_center + max_axis_width / 2) + max_axis_width * CONFIG["padding"])
    y_center = (result.y_max + result.y_min) / 2
    y_range = ((y_center - max_axis_width / 2) - max_axis_width * CONFIG["padding"],
               (y_center + max_axis_width / 2) + max_axis_width * CONFIG["padding"])

    # Graph points
    plt.figure(dpi=1000, figsize=(10, 10))
    # fig = plt.figure(figsize=(200,200))
    ax = plt.subplot(111, aspect='equal')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.plot(result.x, result.y, linewidth=CONFIG["line_width"], color=CONFIG["color"])

    # This is the code to do gradient lines. Warning: it's slow as shit
    # ==================================================================
    # cm = plt.get_cmap("twilight_shifted")
    # cycle = [cm(i / (len(result) - 1)) for i in range(len(result) - 1)]
    # ax.set_prop_cycle(color=cycle)
    # for i in range(len(result)):
    #     ax.plot(result.x[i:i+2], result.y[i:i+2], linewidth=CONFIG["line_width"])

    ax.set(xlim=x_range, ylim=y_range)
    plt.axis("off")
    plt.savefig(CONFIG["output_file"])
    plotted = datetime.now()
    print(f"PLOT RENDERED IN {(plotted - points).total_seconds()} SECONDS "
          f"(TOTAL: {(plotted - start).total_seconds()} SECONDS)")


CONFIG = {
    "DEBUG": False,
    "VERBOSE": False,
    "PRINT_RESULTS": False,

    "time_step": 0.1 / 4,
    "abort_on_cycle": True,
    "smoothing": False,  # Interpolates points if the pen moved a lot between them.
                        # Slows renders, but produces better results on large time steps
    "max_point_sep": 1,    # The distance above which adjacent points will be smoothed
    "num_points": 6000000,
    "padding": 0.1,
    "line_width": 0.2,
    "output_file": "graph.pdf",
    "color": "black",
    "fl_precision": 0.000000001,
}


if __name__ == '__main__':
    if CONFIG["time_step"] <= 0:
        print("-----------------------------------------------------\n"
              "ERROR: Time increment must be a positive number"
              "\n-----------------------------------------------------")
        exit(-1)

    # Rotor params
    p = {
        "radius_a": 10,
        "theta_a": 90,
        "omega_a": -10.1,
        "length_a": 15,

        "radius_b": 10,
        "theta_b": 0,
        "omega_b": 10,
        "length_b": 15,

        "rotor_sep": 8,

        "world_x_coord": 5,
        "world_y_coord": 30,
        "world_omega": 0,
    }
    main(**p)
