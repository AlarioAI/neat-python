import argparse
import enum
import functools
import os
import neat
import copy
import math
import time
import multiprocessing
import queue
import sys
import tqdm # Assume tqdm is installed

# Assuming genome_srl.py is in the same directory or accessible in PYTHONPATH
try:
    from genome_srl import serialize_genome
except ImportError:
    print("Error: genome_srl.py not found. Please ensure it's in the Python path.")
    print("Genome serialization/saving will be disabled.")
    serialize_genome = None # Fallback

# --- Enum for Operations ---
class BinaryOperation(enum.Enum):
    XOR = "xor"
    XNOR = "xnor"
    A_NIMP_B = "a_nimp_b" # A AND NOT B
    B_NIMP_A = "b_nimp_a" # B AND NOT A

    def __str__(self):
        return self.value

# --- Input and Output Definitions ---
COMMON_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

OPERATIONS_DATA = {
    BinaryOperation.XOR: {
        "outputs": [[0.0], [1.0], [1.0], [0.0]],
        "name": "XOR"
    },
    BinaryOperation.XNOR: {
        "outputs": [[1.0], [0.0], [0.0], [1.0]],
        "name": "XNOR (Equivalence)"
    },
    BinaryOperation.A_NIMP_B: {
        "outputs": [[0.0], [0.0], [1.0], [0.0]],
        "name": "A ⇏ B (A Not Implies B / A AND NOT B)"
    },
    BinaryOperation.B_NIMP_A: {
        "outputs": [[0.0], [1.0], [0.0], [0.0]],
        "name": "B ⇏ A (B Not Implies A / B AND NOT A)"
    }
}

# --- Attribute Truncation Function (Unchanged) ---
def truncate_genome_float_attributes(original_genome: neat.genome.DefaultGenome,
                                      neat_config: neat.config.Config,
                                      decimal_places: int) -> neat.genome.DefaultGenome:
    if not isinstance(decimal_places, int):
        raise TypeError(f"decimal_places must be an integer, got {type(decimal_places)}.")
    if decimal_places < 0:
        raise ValueError(f"decimal_places must be non-negative, got {decimal_places}.")

    copied_genome = copy.deepcopy(original_genome)
    genome_config = neat_config.genome_config
    factor = 10 ** decimal_places

    min_weight_limit = genome_config.weight_min_value
    max_weight_limit = genome_config.weight_max_value
    for _, connection in copied_genome.connections.items():
        original_weight = connection.weight
        truncated_weight = math.trunc(original_weight * factor) / factor
        connection.weight = max(min_weight_limit, min(max_weight_limit, truncated_weight))

    min_bias_limit = genome_config.bias_min_value
    max_bias_limit = genome_config.bias_max_value
    min_response_limit = genome_config.response_min_value
    max_response_limit = genome_config.response_max_value
    for _, node in copied_genome.nodes.items():
        original_bias = node.bias
        truncated_bias = math.trunc(original_bias * factor) / factor
        node.bias = max(min_bias_limit, min(max_bias_limit, truncated_bias))

        original_response = node.response
        truncated_response = math.trunc(original_response * factor) / factor
        node.response = max(min_response_limit, min(max_response_limit, truncated_response))

    return copied_genome

# --- Fitness Evaluation Function (Unchanged) ---
def eval_genomes_op(genomes, config, current_op_inputs, current_op_outputs):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(current_op_inputs, current_op_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

# --- Helper function to run NEAT in a separate process (Unchanged) ---
def run_neat_process(config, eval_func, result_queue):
    try:
        p = neat.Population(config)
        winner = p.run(eval_func)
        result_queue.put(winner)
    except neat.CompleteExtinctionException as e:
        result_queue.put(e)
    except Exception as e:
        result_queue.put(e)

# --- Function to Find One Solution (Unchanged) ---
def find_one_solution(neat_config, op_enum_member, timeout_seconds):
    op_data = OPERATIONS_DATA[op_enum_member]
    op_inputs = op_data.get("inputs", COMMON_INPUTS)
    op_outputs = op_data["outputs"]

    eval_function_for_run = functools.partial(eval_genomes_op,
                                              current_op_inputs=op_inputs,
                                              current_op_outputs=op_outputs)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_neat_process,
        args=(neat_config, eval_function_for_run, result_queue)
    )

    process.start()
    process.join(timeout=timeout_seconds)

    winner_genome = None
    if process.is_alive():
        print("  Timeout: Terminating run.", end="")
        process.terminate()
        time.sleep(0.1)
        if process.is_alive():
             if hasattr(process, 'kill'): process.kill()
             else: os.kill(process.pid, 9)
        process.join()
        print(" Terminated.")
        winner_genome = None
    else:
        if process.exitcode == 0:
            try:
                result = result_queue.get_nowait()
                if isinstance(result, neat.genome.DefaultGenome):
                    print("  Success: Solution found.", end="")
                    winner_genome = result
                elif isinstance(result, neat.CompleteExtinctionException):
                    print("  Failure: Complete extinction.", end="")
                    winner_genome = None
                elif isinstance(result, Exception):
                     print(f"  Failure: Run failed with exception: {result}", end="")
                     winner_genome = None
                else:
                     print(f"  Failure: Unexpected result type {type(result)}.", end="")
                     winner_genome = None
            except queue.Empty:
                print("  Failure: Process finished but no result found.", end="")
                winner_genome = None
            except Exception as e:
                print(f"  Failure: Error retrieving result: {e}", end="")
                winner_genome = None
        else:
            print(f"  Failure: Process terminated with exit code {process.exitcode}.", end="")
            winner_genome = None
    print() # Newline after status message
    try:
       while not result_queue.empty():
           result_queue.get_nowait()
    except Exception:
        pass
    result_queue.close()
    # result_queue.join_thread() # Typically not needed with standard Queue unless daemon threads used

    return winner_genome


# --- Main Script Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate N solutions for binary logical operations using NEAT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=[op.value for op in BinaryOperation] + ["all"],
        default="xor",
        help="Logical operation to evolve or 'all' for all functions."
    )
    parser.add_argument(
        "-n", "--num_solutions",
        type=int,
        default=1,
        metavar="N",
        help="Number of solutions (independent evolution runs) to generate per operation."
    )
    parser.add_argument(
        "--solutions_dir",
        type=str,
        default="solutions",
        metavar="PATH",
        help="Directory where solution subdirectories will be created."
    )
    parser.add_argument(
        "--truncate_weights_dp",
        type=int,
        default=2,
        metavar="N",
        help="Decimal places (N >= 0) for truncating weights, biases, responses. Negative disables."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Time limit in seconds for each individual solution run."
    )
    args = parser.parse_args()

    if args.num_solutions < 1:
        print("Error: --num_solutions must be at least 1.")
        exit(1)
    if args.timeout <= 0:
        print("Error: --timeout must be positive.")
        exit(1)

    # Find config file
    local_dir = os.path.dirname(__file__)
    if not local_dir: local_dir = "."
    config_path = os.path.join(local_dir, 'config-feedforward')
    if not os.path.exists(config_path):
         config_path_alt = os.path.join(os.path.dirname(local_dir), 'xor', 'config-feedforward')
         if os.path.exists(config_path_alt):
             config_path = config_path_alt
         else:
             print(f"Error: Configuration file 'config-feedforward' not found.")
             print(f"  Checked relative paths: '{config_path}', '{config_path_alt}'")
             print("  Place it relative to the script (e.g., in ./ or ./xor/).")
             exit(1)

    print(f"Using configuration file: {os.path.abspath(config_path)}")

    # Load NEAT config
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except Exception as e:
        print(f"Error loading NEAT configuration: {e}")
        exit(1)

    # Determine operations to run
    operations_to_run = []
    if args.operation.lower() == "all":
        operations_to_run = list(BinaryOperation)
    else:
        try:
            operations_to_run.append(BinaryOperation(args.operation.lower()))
        except ValueError:
            print(f"Error: Invalid operation '{args.operation}'.")
            print(f"Choices are: {[op.value for op in BinaryOperation]} or 'all'.")
            exit(1)

    # Create base solutions directory
    try:
        os.makedirs(args.solutions_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating solutions directory '{args.solutions_dir}': {e}")
        exit(1)

    # --- Generation Loop ---
    total_solutions_generated = 0
    total_runs_attempted = 0

    for op_enum in operations_to_run:
        op_key = str(op_enum)
        op_name_friendly = OPERATIONS_DATA[op_enum]["name"]
        op_solutions_dir = os.path.join(args.solutions_dir, op_key)

        try:
            os.makedirs(op_solutions_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory '{op_solutions_dir}': {e}")
            continue # Skip this operation

        print(f"\n--- Generating {args.num_solutions} solutions for {op_name_friendly} ---")
        solutions_found_for_op = 0

        # Use tqdm.trange directly, assuming it's installed
        pbar = tqdm.trange(args.num_solutions, desc=f"{op_name_friendly}", ncols=100, unit="run")
        for i in pbar:
            run_index = i + 1
            total_runs_attempted += 1
            # Update description for current run - using set_postfix might be better
            # pbar.set_description(f"{op_name_friendly} [Run {run_index}/{args.num_solutions}]")
            pbar.set_postfix_str(f"attempt {run_index}/{args.num_solutions}, saved: {solutions_found_for_op}", refresh=True)


            winner = find_one_solution(config, op_enum, args.timeout)

            if winner:
                genome_to_save = winner
                if args.truncate_weights_dp >= 0:
                    try:
                        genome_to_save = truncate_genome_float_attributes(
                            winner, config, args.truncate_weights_dp
                        )
                    except Exception as e:
                        print(f"\nWarning: Failed to truncate genome for solution run {run_index}: {e}")

                if serialize_genome:
                    try:
                        serialized_string = serialize_genome(op_key, genome_to_save, config, args.truncate_weights_dp)
                        filename = os.path.join(op_solutions_dir, f"sol_{solutions_found_for_op}.genome.txt")
                        with open(filename, 'w') as f:
                            f.write(serialized_string)
                        solutions_found_for_op += 1
                    except Exception as e:
                        print(f"\nError saving solution from run {run_index} to file: {e}")
                else:
                    print(f"\nWarning: serialize_genome not available. Cannot save solution from run {run_index}.")
                    break # Stop trying if saving isn't possible

            # Update postfix again after potential success/failure
            pbar.set_postfix_str(f"attempt {run_index}/{args.num_solutions}, saved: {solutions_found_for_op}", refresh=True)


        pbar.close() # Ensure the progress bar finishes cleanly
        print(f"--- Finished {op_name_friendly}: Found and saved {solutions_found_for_op} / {args.num_solutions} solutions in '{op_solutions_dir}' ---")
        total_solutions_generated += solutions_found_for_op

    print(f"\n=== Generation Complete ===")
    print(f"Attempted {total_runs_attempted} runs across all operations.")
    print(f"Successfully generated and saved {total_solutions_generated} solution files in '{args.solutions_dir}'.")

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()

    # Optional: Set start method (consider potential side effects)
    # try:
    #    if multiprocessing.get_start_method() != 'spawn':
    #        multiprocessing.set_start_method('spawn', force=True)
    # except Exception as e:
    #    print(f"Note: Could not set multiprocessing start method to 'spawn': {e}")

    main()