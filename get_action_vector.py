from gym_problem import action_mapping
import numpy as np

def get_action_vector(mapped_value):
    # Reverse the action_mapping to map from mapped value back to binary value
    reverse_mapping = {v: k for k, v in action_mapping.items()}
    
    # Retrieve the binary value
    binary_value = reverse_mapping.get(mapped_value)
    if binary_value is None:
        raise ValueError(f"Mapped value {mapped_value} not found in mapping.")
    
    binary_vector = np.array([int(bit) for bit in format(binary_value, '07b')])
    return binary_vector

