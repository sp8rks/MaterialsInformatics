import re


def remove_string_braket(value_string):
        """
        Removes parentheses from a value string and convert to float if possible.
        """
        return float(value_string.split('(')[0]) if '(' in value_string else float(value_string)
