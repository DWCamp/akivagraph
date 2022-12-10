"""
Defines the classes needed for the spirograph
"""
import matplotlib.pyplot as plt
import os
import time

from config import CONFIG
from pointcompute import PointCompute
from rotor import Rotor
from utils import report


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
    start_t = time.perf_counter()
    smoothing = CONFIG["max_point_sep"] if CONFIG["smoothing"] else None
    result = PointCompute(rotor_a, rotor_b, world, CONFIG["step_length"], CONFIG["max_steps"], smoothing)
    points_t = time.perf_counter()
    if CONFIG["print_points"]:
        for x, y in result:
            print(f"{x}\t{y}")
    print(f"{len(result)} POINTS GENERATED IN {round(points_t - start_t, 7)} SECONDS")

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
    plot_t = time.perf_counter()
    print(f"PLOT RENDERED IN {round(plot_t - points_t, 7)} SECONDS")
    print(f"-- TOTAL: {round(plot_t - start_t, 7)} SECONDS")
    print(f"\nOutput file: {output_file}")

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
        "world_omega": 0.27,
    }
    main(**p)
