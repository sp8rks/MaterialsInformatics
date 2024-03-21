import gemmi
import re
from util.string_parser import remove_string_braket
import numpy as np
from fractions import Fraction


def get_atom_type(label):
    # Splitting the label into separate parts if it contains parentheses
    parts = re.split(r'[()]', label)
    for part in parts:
        # Attempt to extract the atom type
        match = re.search(r'([A-Z][a-z]*)', part)
        if match:
            return match.group(1)
    return None



def extract_formula_and_atoms(block):
    """
    Extract the chemical formula and unique atoms from a CIF block.
    """

    A_labels = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Th", "U"]
    B_labels = ["Si", "Ga", "Ge", "In", "Sn", "Sb", "Al"]
    M_labels = ["Fe", "Co", "Ni", "Ru", "Rh", "Pd", "Os", "Ir", "Pt"]
    
    formula = None
    formula_string = None

    if block.find_pair('_chemical_formula_structural'):
        formula = block.find_pair('_chemical_formula_structural')
    elif block.find_pair('_chemical_formula_sum'):
        formula = block.find_pair('_chemical_formula_sum')
    
    if formula:
        formula_string = formula[1]
        formula_string = formula_string.replace("'", "")
        formula_string = re.sub('[~ ]', '', formula_string)

    if not formula_string:
        return None, 0, None

    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    matches = pattern.findall(formula_string)
    unique_atoms_tuple = [(atom, int(count) if count else 1) for atom, count in matches]
    num_of_unique_atoms = len({atom for atom, _ in unique_atoms_tuple})
    sorted_unique_atoms_tuple = None

    A_elements = []
    B_elements = []
    M_elements = []

    for atom, count in unique_atoms_tuple:
        if atom in A_labels:
            A_elements.append((atom, count))
        elif atom in B_labels:
            B_elements.append((atom, count))
        elif atom in M_labels:
            M_elements.append((atom, count))


    # Binary system
    if num_of_unique_atoms == 2:
        if len(A_elements) == 0 and len(B_elements) == 0:
            sorted_unique_atoms_tuple = unique_atoms_tuple
        elif len(A_elements) == 0 and len(B_elements) == 1:
            A_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in B_labels]
        elif len(B_elements) == 0 and len(A_elements) == 1:
            B_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in A_labels]
        
        if len(A_elements) != 0 or len(B_elements) != 0:
            sorted_unique_atoms_tuple = A_elements + B_elements

    # Ternary system
    if num_of_unique_atoms == 3:
        if len(A_elements) == 0 and len(B_elements) == 0 and len(M_elements) == 0:
            sorted_unique_atoms_tuple = unique_atoms_tuple
        elif len(A_elements) == 1 and len(B_elements) == 1:
            # One atom each in A and B: the remaining one should be M
            M_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in A_labels and atom not in B_labels]
        elif len(A_elements) == 0 and len(B_elements) == 1:
            # One atom in B: the remaining ones should be A and M
            M_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in B_labels]
            A_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in B_labels and atom not in M_labels]
        elif len(A_elements) == 1 and len(B_elements) == 0:
            # One atom in A: the remaining ones should be B and M
            M_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in A_labels]
            B_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in A_labels and atom not in M_labels]
        elif len(A_elements) == 1 and len(M_elements) == 1:
            # One atom in A and one in M: the remaining one should be B
            B_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in A_labels and atom not in M_labels]
        elif len(B_elements) == 1 and len(M_elements) == 1:
            # One atom in B and one in M: the remaining one should be A
            A_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in B_labels and atom not in M_labels]
        elif len(A_elements) == 0 and len(M_elements) == 1:
            # One atom in M: the remaining ones should be A and B
            A_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in B_labels and atom not in M_labels]
            B_elements = [(atom, count) for atom, count in unique_atoms_tuple if atom not in A_labels and atom not in M_labels]
        
        if len(A_elements) != 0 or len(B_elements) != 0 or len(M_elements) != 0:
            sorted_unique_atoms_tuple = A_elements + M_elements + B_elements

    return sorted_unique_atoms_tuple, num_of_unique_atoms, formula_string



def get_loop_tags():
    """
    Returns a list of predefined loop tags commonly used for atomic site description.
    """
    loop_tags = ["_atom_site_label", "_atom_site_type_symbol",
            "_atom_site_symmetry_multiplicity", "_atom_site_Wyckoff_symbol",
            "_atom_site_fract_x", "_atom_site_fract_y","_atom_site_fract_z", "_atom_site_occupancy"]

    return loop_tags


def get_unit_cell_lengths_angles(block):
    """
    Returns the unit cell lengths and angles from a given block.
    """
    keys_lengths = ['_cell_length_a', '_cell_length_b', '_cell_length_c']
    keys_angles = ['_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']

    lengths = [remove_string_braket(block.find_value(key)) for key in keys_lengths]
    angles = [remove_string_braket(block.find_value(key)) for key in keys_angles]

    return tuple(lengths + angles)


def exceeds_atom_count_limit(all_points, max_atoms_count):
    """
    Checks if the number of unique atomic positions after applying symmetry operations 
    exceeds the specified atom count limit.
    """

    return len(all_points) > max_atoms_count


def get_CIF_block(filename):
    """
    Returns a CIF block from its CIF filename.
    """
    doc = gemmi.cif.read_file(filename)
    block = doc.sole_block()
    
    return block


def get_loop_values(block, loop_tags):
    """
    Retrieves loop values from a block for the specified tags.
    """
    loop_values = [block.find_loop(tag) for tag in loop_tags]

    # Check for zero or missing coordinates
    if len(loop_values[4]) == 0 or len(loop_values[5]) == 0 or len(loop_values[6]) == 0:  # missing coordinates
        raise RuntimeError("Missing atomic coordinates")
        
    # print("loop_values:", loop_values)
    return loop_values


def print_loop_values(loop_values, i):
    """
    Prints the loop values for a specific index with a descriptive format.
    """
    descriptions = [
        "Atom Site Label:",
        "Atom Site Type Symbol:",
        "Atom Site Symmetry Multiplicity:",
        "Atom Site Wyckoff Symbol:",
        "Atom Site Fract X:",
        "Atom Site Fract Y:",
        "Atom Site Fract Z:",
        "Atom Site Occupancy:"
    ]
    
    for idx, desc in enumerate(descriptions):
        value = loop_values[idx][i]
        if "Fract" in desc:
            value = float(value)
        elif "Symmetry Multiplicity" in desc:
            value = int(value)
        print(f"{desc} {value}")
    print()

#==============================================
def convert_decimal_to_fraction(decimal_str):
    """
    Convert a decimal string to a fraction string with a certain level of precision.
    """
    fraction = Fraction(float(decimal_str)).limit_denominator(1000)
    return str(fraction)

def preprocess_cif_file(cif_file_path):
    """
    Preprocesses a CIF file, converting decimal numbers in the symmetry operations to fractions.
    Example:
    '1/3+y, 2/3+x, 1.16667-z'
    
    To
    
    '1/3+y, 2/3+x, 7/6-z'
    """
    with open(cif_file_path, 'r') as file:
        lines = file.readlines()

    decimal_regex = re.compile(r'\d+\.\d+')

    for i, line in enumerate(lines):
        if '_space_group_symop_operation_xyz' in line:
            while True:
                i += 1
                if 'loop_' in lines[i] or i >= len(lines):
                    break

                if re.findall("^ \d+ ", lines[i]):
                    lines[i] = decimal_regex.sub(lambda m: convert_decimal_to_fraction(m.group()), lines[i])

    with open(cif_file_path, 'w') as file:
        file.writelines(lines)
        
        
def take_care_of_atomic_site(cif_file_path):
    """
    Preprocesses a CIF file to make the atom symbols right for further parsing.
    Example:
    M1 Al - - - - -
    M2 Al - - - - -
    M3 Al - - - - -
    M2 Sb - - - - -
    
    To
    
    Al1 Al - - - - -
    Al2 Al - - - - -
    Al3 Al - - - - -
    Sb1 Sb - - - - -
    """
    with open(cif_file_path, 'r') as file:
        lines = file.readlines()

    atom_symbol_count = {}  # Dictionary to keep track of atom symbol counts
    process_lines = False
    for i, line in enumerate(lines):
        if re.findall("^ _atom_site", line):
            process_lines = True
        elif process_lines:
            if line.strip():  # Check if the line is not blank
                parts = line.split()
                if len(parts) > 0:
                    atom_symbol = parts[1]
                    atom_symbol_count[atom_symbol] = atom_symbol_count.get(atom_symbol, 0) + 1
                    parts[0] = atom_symbol + str(atom_symbol_count[atom_symbol])
                    lines[i] = ' '+' '.join(parts) + '\n'
            else:
                process_lines = False

    with open(cif_file_path, 'w') as file:
        file.writelines(lines)

def valid_cif(cif_file_path):
    with open(cif_file_path, "r") as file:
        lines = file.readlines()

    is_valid = False
    for i in lines:
        if get_loop_tags()[1] in i:
            is_valid = True
            break

    return is_valid