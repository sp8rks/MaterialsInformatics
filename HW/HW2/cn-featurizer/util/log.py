def print_json_pretty(data_name, data):
    """
    Prints the contents of a dictionary in a pretty format.
    """
    print(f"\nData from {data_name}:")
    for key, value in data.items():
        if isinstance(value, (list, str)):
            print(f"- {key}: {value[0]}")
        else:
            print(f"- {key}: {value}")
    print()


