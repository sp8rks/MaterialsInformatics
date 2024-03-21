import numpy as np


def get_radians_from_degrees(angles):
    """
    Converts angles from degrees to radians and round to 5 decimal places.
    """
    radians = [round(np.radians(angle), 5) for angle in angles]

    return radians


def get_cartesian_from_fractional(frac_coords, cell_lengths, cell_angles):
    """
    Convert fractional coordinates to Cartesian coordinates based on given cell lengths and angles.
    """
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles

    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    volume = 1.0 - cosa**2.0 - cosb**2.0 - cosg**2.0 + 2.0 * cosa * cosb * cosg
    volume = np.sqrt(volume)
    r = np.zeros((3, 3))
    r[0, 0] = a
    r[0, 1] = b * cosg
    r[0, 2] = c * cosb
    r[1, 1] = b * sing
    r[1, 2] = c * (cosa - cosb * cosg) / sing
    r[2, 2] = c * volume / sing

    # Convert fractional coordinates to Cartesian coordinates
    cart_coords = np.dot(r, frac_coords)

    return cart_coords


def rounded_distance(distance, precision=2):
    """
    Round a distance value to a specified precision.
    """
    
    return round(distance, precision)


