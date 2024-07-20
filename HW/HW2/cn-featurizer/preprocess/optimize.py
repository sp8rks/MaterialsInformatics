from functools import partial
from scipy.optimize import minimize

def objective_binary(params, A_CIF_rad, B_CIF_rad):
    """
    Calculates the objective value for binary systems by computing the sum of squared percent differences 
    between original and refined CIF radii for two atoms.
    """
    A_CIF_rad_refined, B_CIF_rad_refined = params

    # Calculate differences between original and refined radii
    A_CIF_rad_diff = A_CIF_rad - A_CIF_rad_refined
    B_CIF_rad_diff = B_CIF_rad - B_CIF_rad_refined

    # Calculate percent differences
    A_CIF_rad_diff_percent = A_CIF_rad_diff / A_CIF_rad
    B_CIF_rad_diff_percent = B_CIF_rad_diff / B_CIF_rad

    # Square the percent differences
    A_CIF_rad_diff_percent_squared = A_CIF_rad_diff_percent**2
    B_CIF_rad_diff_percent_squared = B_CIF_rad_diff_percent**2

    # Return the sum of squared percent differences
    return A_CIF_rad_diff_percent_squared + B_CIF_rad_diff_percent_squared


def objective_ternary(params, R_CIF_rad, M_CIF_rad, X_CIF_rad):
    """
    Calculates the objective value for ternary systems by computing the sum of squared percent differences 
    between original and refined CIF radii for three atoms.
    """
    R_CIF_rad_refined, M_CIF_rad_refined, X_CIF_rad_refined = params

    # Calculate differences between original and refined radii
    R_CIF_rad_diff = R_CIF_rad - R_CIF_rad_refined
    M_CIF_rad_diff = M_CIF_rad - M_CIF_rad_refined
    X_CIF_rad_diff = X_CIF_rad - X_CIF_rad_refined

    # Calculate percent differences
    R_CIF_rad_diff_percent = R_CIF_rad_diff / R_CIF_rad
    M_CIF_rad_diff_percent = M_CIF_rad_diff / M_CIF_rad
    X_CIF_rad_diff_percent = X_CIF_rad_diff / X_CIF_rad

    # Square the percent differences
    R_CIF_rad_diff_percent_squared = R_CIF_rad_diff_percent**2
    M_CIF_rad_diff_percent_squared = M_CIF_rad_diff_percent**2
    X_CIF_rad_diff_percent_squared = X_CIF_rad_diff_percent**2

    # Return the sum of squared percent differences
    return R_CIF_rad_diff_percent_squared + M_CIF_rad_diff_percent_squared + X_CIF_rad_diff_percent_squared


def constraint_binary_1(params, shortest_AA):
    A_CIF_rad_refined, B_CIF_rad_refined = params
    return shortest_AA - (2 * A_CIF_rad_refined)


def constraint_binary_2(params, shortest_BB):
    A_CIF_rad_refined, B_CIF_rad_refined = params
    return shortest_BB - (2 * B_CIF_rad_refined)


def constraint_binary_3(params, shortest_AB):
    A_CIF_rad_refined, B_CIF_rad_refined = params
    return shortest_AB - (A_CIF_rad_refined + B_CIF_rad_refined)


def constraint_ternary(params, shortest_distance, labels):
    # Assuming labels will be something like "RR", "MX", etc.
    multipliers = {
        "RR": (2, 0, 0),
        "MM": (0, 2, 0),
        "XX": (0, 0, 2),
        "RM": (1, 1, 0),
        "MX": (0, 1, 1),
        "RX": (1, 0, 1),
    }
    
    multiplier = multipliers[labels]
    sum_refined = sum(m*p for m, p in zip(multiplier, params))
    return shortest_distance - sum_refined


def optimize_CIF_rad_binary(A_CIF_rad, B_CIF_rad, shortest_distances_pair, return_obj_value=False):
    """
    Optimizes CIF radii for a binary system, given the initial radii and shortest distances between atom pairs. 
    It sets up the constraints based on the shortest distances and employs the minimizer to optimize the radii.
    """

    # Construct constraint dictionaries
    con1 = {'type': 'eq', 'fun': partial(constraint_binary_1, shortest_AA=shortest_distances_pair['AA'])}
    con2 = {'type': 'eq', 'fun': partial(constraint_binary_2, shortest_BB=shortest_distances_pair['BB'])}
    con3 = {'type': 'eq', 'fun': partial(constraint_binary_3, shortest_AB=shortest_distances_pair['AB'])}

    constraints_AA_AB = [con1, con3]
    constraints_AA_BB = [con1, con2]
    constraints_AB_AA = [con3, con1]
    constraints_AB_BB = [con3, con2]
    constraints_BB_AA = [con2, con1]
    constraints_BB_AB = [con2, con3]

    constraint_mapping = {
        ("AA", "AB"): constraints_AA_AB, ("AA", "BB"): constraints_AA_BB, 
        ("AB", "AA"): constraints_AB_AA, ("AB", "BB"): constraints_AB_BB, 
        ("BB", "AA"): constraints_BB_AA, ("BB", "AB"): constraints_BB_AB
    }

    # Sort distances
    sorted_distances = sorted(shortest_distances_pair.items(), key=lambda x: x[1])
    
    # Extract shortest pairs
    first_shortest_pair = sorted_distances[0][0]
    second_shortest_pair = sorted_distances[1][0]


    # Define the initial guess for the minimizer
    init_guess = [A_CIF_rad, B_CIF_rad]

    result = None
    pair = (first_shortest_pair, second_shortest_pair)
    if pair in constraint_mapping:
        objective_func = partial(objective_binary, A_CIF_rad=A_CIF_rad, B_CIF_rad=B_CIF_rad)
        result = minimize(objective_func, init_guess, constraints=constraint_mapping[pair])
    else:
        print(f"No constraints defined for pair {pair}.")

    
    if return_obj_value:
        return result.x, result.fun
    else:
        return result.x


def optimize_CIF_rad_ternary(R_CIF_rad, M_CIF_rad, X_CIF_rad, shortest_distances_pair, return_obj_value=False):
    """
    Optimizes CIF radii for a ternary system, given the initial radii and shortest distances between atom pairs. 
    It sets up the constraints based on the shortest distances and employs the minimizer to optimize the radii.
    """
    # Create generic constraints based on shortest distances
    constraints = {}
    for label, dist in shortest_distances_pair.items():
        constraints[label] = {'type': 'eq', 'fun': partial(constraint_ternary, shortest_distance=dist, labels=label)}

    # Map these constraints to pairings
    constraint_mapping = {}
    labels = list(shortest_distances_pair.keys())
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            constraint_mapping[(label1, label2)] = [constraints[label1], constraints[label2]]
            constraint_mapping[(label2, label1)] = [constraints[label2], constraints[label1]]
    
    # Sort distances
    sorted_distances = sorted(shortest_distances_pair.items(), key=lambda x: x[1])

    # Extract shortest pairs
    first_shortest_pair = sorted_distances[0][0]
    second_shortest_pair = sorted_distances[1][0]

    # Define the initial guess for the minimizer
    init_guess = [R_CIF_rad, M_CIF_rad, X_CIF_rad]

    result = None
    pair = (first_shortest_pair, second_shortest_pair)
    if pair in constraint_mapping:
        objective_func = partial(objective_ternary, R_CIF_rad=R_CIF_rad, M_CIF_rad=M_CIF_rad, X_CIF_rad=X_CIF_rad)
        result = minimize(objective_func, init_guess, constraints=constraint_mapping[pair])
    else:
        print(f"No constraints defined for pair {pair}.")

    if return_obj_value:
        return result.x, result.fun
    else:
        return result.x



