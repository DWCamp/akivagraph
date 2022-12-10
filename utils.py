"""
Misc. functions which are helpful to the script as a whole but do not belong to any one file in particular

@author - D. William Campman
@date - 2022.09.12
"""
import math
from typing import Optional
from config import CONFIG


def dist(a, b):
    """
    Computes the Euclidean distance between two points
    :param a: The (x, y) coordinates of point a
    :param b: The (x, y) coordinates of point b
    :return: The Euclidean distance between them
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def fl_eq(f1: float, f2: float, mod: Optional[float] = None):
    """
    Checks if two floating point numbers are equal within the margin of error
    specified in the config file, in the event floating point math errors may
    cause two numbers to differ slightly that should be equivalent
    :param f1: The first number
    :param f2: The second number
    :param mod: This value specifies if the check needs to be performed across
    a modulo boundary (e.g. in a circle, 0° ~= 359.99999999°)
    :return: `True` if the difference between values is less than the precision
    """
    flp = CONFIG["fl_precision"]
    if mod is not None:
        return abs((f1 - f2 + flp) % mod) < flp * 2
    return abs(f1 - f2) < flp


def report(msg: str, lpad: str = " ", abort: bool = False) -> None:
    """
    Prints a formatted warning to the console
    :param msg: The alert to print
    :param lpad: A string to pad each line with on the left (Default: ' ')
    :param abort: Exit the program after printing the message
    """
    if CONFIG["suppress_report"]:
        if abort:
            exit(-1)
        return

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
