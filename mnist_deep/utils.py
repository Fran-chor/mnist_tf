import tensorflow as tf


def print_white():
    tf.print('\033[97m', end="")


def print_reset_color():
    tf.print("\033[0m", end="")


# To print in white with tensorflow, tf.print is else in red
def tfw_print(*args):
    print_white()
    tf.print(*args)
    print_reset_color()
