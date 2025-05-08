import argparse
import os
import neat
from neat.nn import FeedForwardNetwork, RecurrentNetwork

# Assuming genome_srl.py is in the same directory or accessible in PYTHONPATH
# It contains the deserialize_genome function and serialization constants.
try:
    from genome_srl import deserialize_genome, S_GEN_START
except ImportError:
    print("Error: genome_srl.py not found. Please ensure it's in the Python path.")
    exit(1)

# --- Data for known operations (similar to binary_ops.py) ---
# Inputs for 2-input binary operations
BINARY_OP_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

# Expected outputs and names for the operations
# The keys (e.g., "xor") should match the op_name in the serialized genome file.
OPERATIONS_DATA = {
    "xor": {
        "inputs": BINARY_OP_INPUTS,
        "outputs": [[0.0], [1.0], [1.0], [0.0]],
        "name": "XOR"
    },
    "xnor": {
        "inputs": BINARY_OP_INPUTS,
        "outputs": [[1.0], [0.0], [0.0], [1.0]],
        "name": "XNOR (Equivalence)"
    },
    "a_nimp_b": { # A Not Implies B (A AND NOT B)
        "inputs": BINARY_OP_INPUTS,
        "outputs": [[0.0], [0.0], [1.0], [0.0]],
        "name": "A NIMP B (A and Not B)"
    },
    "b_nimp_a": { # B Not Implies A (B AND NOT A)
        "inputs": BINARY_OP_INPUTS,
        "outputs": [[0.0], [1.0], [0.0], [0.0]],
        "name": "B NIMP A (B and Not A)"
    },
    # Add other operations here if needed, for example:
    # "and": {
    #     "inputs": BINARY_OP_INPUTS,
    #     "outputs": [[0.0], [0.0], [0.0], [1.0]],
    #     "name": "AND"
    # },
    # "or": {
    #     "inputs": BINARY_OP_INPUTS,
    #     "outputs": [[0.0], [1.0], [1.0], [1.0]],
    #     "name": "OR"
    # },
}

def extract_op_name_from_string(genome_string):
    """
    Extracts the operation name from the first line of the serialized genome string.
    Example line: _S_GEN_ </xor>
    """
    first_line = genome_string.split('\n', 1)[0].strip()
    if not first_line.startswith(S_GEN_START + " </"):
        raise ValueError("Serialized genome string does not start with the expected format for op_name.")
    try:
        op_name = first_line.split("</", 1)[1].split(">", 1)[0]
        return op_name
    except IndexError:
        raise ValueError("Could not parse op_name from the first line of the genome string.")

def execute_genome(config_path, genome_file_path):
    """
    Loads a NEAT config and a serialized genome, deserializes the genome,
    creates the network, and executes it on predefined inputs for the
    specified operation, showing the results.
    """
    # 1. Load NEAT Config
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except Exception as e:
        print(f"Error loading NEAT configuration from '{config_path}': {e}")
        return

    # 2. Read Serialized Genome String
    try:
        with open(genome_file_path, 'r') as f:
            genome_string = f.read()
    except FileNotFoundError:
        print(f"Error: Genome file '{genome_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading genome file '{genome_file_path}': {e}")
        return

    # 3. Extract Operation Name
    try:
        op_name = extract_op_name_from_string(genome_string)
        print(f"Operation Name: {op_name}")
    except ValueError as e:
        print(f"Error extracting operation name: {e}")
        return

    # 4. Deserialize Genome
    try:
        genome = deserialize_genome(genome_string, config)
    except ValueError as e:
        print(f"Error deserializing genome: {e}")
        print("Please ensure the genome string format is correct and compatible with the NEAT config.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during genome deserialization: {e}")
        return

    # 5. Get Operation Data (Inputs/Expected Outputs)
    operation_data = OPERATIONS_DATA.get(op_name.lower()) # Use lower for case-insensitivity
    if not operation_data:
        print(f"Error: Operation '{op_name}' is not defined in OPERATIONS_DATA.")
        print(f"Available operations: {', '.join(OPERATIONS_DATA.keys())}")
        return

    inputs = operation_data["inputs"]
    expected_outputs = operation_data["outputs"]
    friendly_op_name = operation_data["name"]
    print(f"Executing for operation: {friendly_op_name}")

    # 6. Create Neural Network
    try:
        if config.genome_config.feed_forward:
            network = FeedForwardNetwork.create(genome, config)
            print("Network type: FeedForwardNetwork")
        else:
            network = RecurrentNetwork.create(genome, config)
            print("Network type: RecurrentNetwork (resetting before each input)")
    except Exception as e:
        print(f"Error creating neural network from genome: {e}")
        return

    # 7. Execute and Display Results
    print("\n--- Results ---")
    total_error = 0.0
    num_cases = len(inputs)

    for i, (xi, xo) in enumerate(zip(inputs, expected_outputs)):
        if not config.genome_config.feed_forward:
            network.reset() # Reset recurrent network for each input sequence if applicable

        actual_output = network.activate(xi)

        # Assuming single output node for these binary operations
        # The output of activate is a list, xo is also a list (e.g., [0.0])
        error = (actual_output[0] - xo[0]) ** 2
        total_error += error

        print(f"  Input: {xi}, Expected: {xo[0]:.4f}, Got: {actual_output[0]:.4f}, Error: {error:.4f}")

    if num_cases > 0:
        mean_squared_error = total_error / num_cases
        print(f"\nMean Squared Error: {mean_squared_error:.4f}")
    print("--- End of Execution ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute a serialized NEAT genome.")
    parser.add_argument("config_path", type=str, help="Path to the NEAT configuration file.")
    parser.add_argument("genome_file_path", type=str, help="Path to the file containing the serialized genome string.")

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: NEAT configuration file '{args.config_path}' not found.")
    elif not os.path.exists(args.genome_file_path):
        print(f"Error: Serialized genome file '{args.genome_file_path}' not found.")
    else:
        execute_genome(args.config_path, args.genome_file_path)