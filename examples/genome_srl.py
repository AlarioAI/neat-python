import neat

# --- Constants for Serialization Format ---
S_GEN_START = "_S_GEN_"
S_GEN_END = "_E_GEN_"
S_CONF_START = "_S_CONF_"
S_CONF_END = "_E_CONF_"
S_NODES_START = "_S_NODES_"
S_NODE_PREFIX = "_NODE_"
S_NODE_END_SUFFIX = "_E_NODES_" # Marks end of one node's definition
S_CONNS_START = "_S_CONNS_"
S_CONN_PREFIX = "_CONN_"
S_CONN_END_SUFFIX = "_E_CONNS_" # Marks end of one connection's definition
S_ATTR_SEP = " "
# FLOAT_PRECISION = 6 # No longer used for fixed precision formatting

# --- Modified Float Formatting ---
def _format_float(value, precision: int):
    """
    Helper to format floats to a string representation without unnecessary
    trailing zeros or decimal points.
    """
    try:
        # Format with the specified precision, then strip trailing zeros and decimal points.
        s = f"{value:.{precision}f}"
        return s
    except Exception:
        raise ValueError("WTF!!!!")

# --- Serialize Genome Function (using the modified _format_float) ---
def serialize_genome(op_name: str, genome: neat.DefaultGenome, neat_config: neat.Config, num_digits: int) -> str:
    """
    Serializes a NEAT genome into a string representation using compact float formatting.
    The format includes:
    - Operation name.
    - Essential configuration (num_inputs, num_outputs, feed_forward, gene types).
    - Node definitions (outputs then renumbered hidden nodes), including bias, response,
      activation, and aggregation. Formats floats compactly.
    - Enabled connection definitions (using renumbered node IDs), including weight.
      Formats floats compactly.

    Args:
        op_name: The name of the operation (e.g., "xor").
        genome: The neat.DefaultGenome object to serialize.
        neat_config: The neat.Config object used for context (num_inputs, etc.).

    Returns:
        A string representation of the genome.
    """
    lines = []
    genome_config = neat_config.genome_config

    # 1. Genome Start and Operation Name
    lines.append(f"{S_GEN_START}{S_ATTR_SEP}</{op_name}>") # Use the format from the user spec

    # 2. Configuration Section
    conf_parts = [
        f"num_inputs{S_ATTR_SEP}{genome_config.num_inputs}",
        f"num_outputs{S_ATTR_SEP}{genome_config.num_outputs}",
        f"feed_forward{S_ATTR_SEP}{genome_config.feed_forward}",
        # Include gene types as requested in the format description
        f"node_gene_type{S_ATTR_SEP}{genome_config.node_gene_type.__name__}",
        f"connection_gene_type{S_ATTR_SEP}{genome_config.connection_gene_type.__name__}"
    ]
    lines.append(f"{S_CONF_START}{S_ATTR_SEP}{S_ATTR_SEP.join(conf_parts)}{S_ATTR_SEP}{S_CONF_END}")

    # 3. Node Renumbering and Serialization
    output_node_ids = set(genome_config.output_keys) # Typically 0 to num_outputs - 1
    # Correctly identify hidden nodes (any node not an input or output)
    # Input nodes are not in genome.nodes; hidden nodes have IDs >= num_outputs IF created sequentially
    # A more robust way might be to get all node keys and subtract output keys.
    # Assuming standard DefaultGenome node ID assignment for hidden nodes starts after outputs:
    original_hidden_node_ids = sorted([nid for nid in genome.nodes if nid not in output_node_ids])

    # Map original hidden node IDs to new sequential IDs starting after max output ID
    hidden_id_map = {orig_id: genome_config.num_outputs + i for i, orig_id in enumerate(original_hidden_node_ids)}

    all_nodes_serialized = []

    # Output Nodes (Sorted by ID: 0 to num_outputs-1)
    sorted_output_node_ids = sorted(list(output_node_ids))
    for node_id in sorted_output_node_ids:
        if node_id in genome.nodes: # Output nodes must exist
            node = genome.nodes[node_id]
            # Use the modified _format_float here
            node_str = S_ATTR_SEP.join([
                str(node_id), # Use original ID for outputs
                _format_float(node.bias, num_digits),
                _format_float(node.response, num_digits),
                node.activation,
                node.aggregation
            ])
            # Use the suffixes from the user spec
            all_nodes_serialized.append(f"{S_NODE_PREFIX}{S_ATTR_SEP}{node_str}{S_ATTR_SEP}{S_NODE_END_SUFFIX}")
        else:
             # This case should ideally not happen with standard NEAT genome creation
             print(f"Warning: Expected output node {node_id} not found in genome.")


    # Hidden Nodes (with renumbered IDs, sorted by NEW ID)
    # Iterate based on the sorted original IDs to ensure consistent mapping order
    for orig_id in original_hidden_node_ids:
        node = genome.nodes[orig_id]
        new_id = hidden_id_map[orig_id]
        # Use the modified _format_float here
        node_str = S_ATTR_SEP.join([
            str(new_id), # Use new renumbered ID
            _format_float(node.bias, num_digits),
            _format_float(node.response, num_digits),
            node.activation,
            node.aggregation
        ])
        # Use the suffixes from the user spec
        all_nodes_serialized.append(f"{S_NODE_PREFIX}{S_ATTR_SEP}{node_str}{S_ATTR_SEP}{S_NODE_END_SUFFIX}")

    if all_nodes_serialized:
        lines.append(S_NODES_START)
        lines.extend(all_nodes_serialized)
        # No explicit end marker for the whole nodes section in the user spec?
        # Let's assume S_CONNS_START implicitly ends nodes if present, or S_GEN_END otherwise.

    # 4. Connections Section (using remapped IDs, only enabled, sorted by remapped IDs)
    connections_serialized = []
    # Sort original connections by key first for determinism before filtering/remapping
    sorted_connections = sorted(list(genome.connections.items()), key=lambda item: item[0])

    for conn_key, conn in sorted_connections:
        if not conn.enabled:
            continue

        from_orig_id, to_orig_id = conn_key

        # Remap from_node_id
        if from_orig_id < 0: # Input node
            remapped_from_id = from_orig_id
        elif from_orig_id in output_node_ids: # Output node (should not happen as source in FF)
            remapped_from_id = from_orig_id # Use original ID
            # Add warning? Feed-forward assumption might be violated.
        else: # Hidden node
            remapped_from_id = hidden_id_map.get(from_orig_id)
            if remapped_from_id is None:
                 # This indicates an issue, maybe a node was deleted but connection remained?
                 print(f"Warning: Connection source node {from_orig_id} has no remapped ID. Skipping connection {conn_key}.")
                 continue

        # Remap to_node_id
        if to_orig_id in output_node_ids: # Output node
            remapped_to_id = to_orig_id # Use original ID
        else: # Hidden node
            remapped_to_id = hidden_id_map.get(to_orig_id)
            if remapped_to_id is None:
                 print(f"Warning: Connection target node {to_orig_id} has no remapped ID. Skipping connection {conn_key}.")
                 continue

        # Use the modified _format_float here
        conn_str_parts = [
            str(remapped_from_id),
            str(remapped_to_id),
            _format_float(conn.weight, num_digits)
        ]
        # Use the suffixes from the user spec
        connections_serialized.append(
            (remapped_from_id, remapped_to_id, f"{S_CONN_PREFIX}{S_ATTR_SEP}{S_ATTR_SEP.join(conn_str_parts)}{S_ATTR_SEP}{S_CONN_END_SUFFIX}")
        )

    # Sort connections based on remapped IDs before adding to lines
    connections_serialized.sort(key=lambda x: (x[0], x[1]))

    if connections_serialized:
        lines.append(S_CONNS_START)
        lines.extend([s for _, _, s in connections_serialized])
        # No explicit end marker for the whole connections section in the user spec?
        # Let's assume S_GEN_END implicitly ends connections.

    # 5. Genome End
    lines.append(S_GEN_END)

    return f"{S_ATTR_SEP}".join(lines)

def deserialize_genome(genome_string: str, neat_config: neat.Config) -> neat.DefaultGenome:
    """
    Deserializes a string representation back into a NEAT DefaultGenome.
    The string format is expected to be a sequence of tokens separated by single spaces.

    Args:
        genome_string: The string representation of the genome.
        neat_config: The neat.Config object to use for creating gene instances
                     and providing context. It's crucial that this config is
                     compatible with the serialized genome's parameters.

    Returns:
        A neat.DefaultGenome object.

    Raises:
        ValueError: If the string format is invalid or inconsistent.
    """
    tokens = genome_string.strip().split(S_ATTR_SEP)
    token_iter = iter(tokens)

    def next_token(expected: str = None) -> str:
        try:
            token = next(token_iter)
            if expected and token != expected:
                raise ValueError(f"Expected token '{expected}', got '{token}'")
            return token
        except StopIteration:
            if expected:
                raise ValueError(f"Unexpected end of string. Expected '{expected}'.")
            else:
                raise ValueError("Unexpected end of string.")

    # 1. Genome Start and Operation Name
    next_token(S_GEN_START)
    op_name_token = next_token() # e.g., </xor>
    # op_name = op_name_token[2:-1] # Extract 'xor' from '</xor>' if needed, but not used for genome reconstruction

    # 2. Configuration Section
    next_token(S_CONF_START)
    parsed_conf = {}
    current_token = next_token()
    while current_token != S_CONF_END:
        key = current_token
        value = next_token()
        parsed_conf[key] = value
        current_token = next_token()
    # S_CONF_END already consumed by the loop condition or last next_token()

    genome_config = neat_config.genome_config
    try:
        s_num_inputs = int(parsed_conf['num_inputs'])
        s_num_outputs = int(parsed_conf['num_outputs'])
        s_feed_forward = parsed_conf['feed_forward'] == 'True'
        s_node_gene_type_name = parsed_conf['node_gene_type']
        s_conn_gene_type_name = parsed_conf['connection_gene_type']
    except KeyError as e:
        raise ValueError(f"Missing essential key in config string: {e}")
    except ValueError as e: # for int conversion
        raise ValueError(f"Invalid value type in config string: {e}")

    # Validate config against neat_config (optional, but good practice)
    if s_num_inputs != genome_config.num_inputs:
        raise ValueError(f"Config mismatch: num_inputs ({s_num_inputs} vs {genome_config.num_inputs})")
    if s_num_outputs != genome_config.num_outputs:
        raise ValueError(f"Config mismatch: num_outputs ({s_num_outputs} vs {genome_config.num_outputs})")
    if s_feed_forward != genome_config.feed_forward:
        raise ValueError(f"Config mismatch: feed_forward ({s_feed_forward} vs {genome_config.feed_forward})")
    if s_node_gene_type_name != genome_config.node_gene_type.__name__:
        raise ValueError(f"Config mismatch: node_gene_type ({s_node_gene_type_name} vs {genome_config.node_gene_type.__name__})")
    if s_conn_gene_type_name != genome_config.connection_gene_type.__name__:
        raise ValueError(f"Config mismatch: connection_gene_type ({s_conn_gene_type_name} vs {genome_config.connection_gene_type.__name__})")

    # Create a new genome. Key can be arbitrary for a standalone genome.
    new_genome_key = 0 # Or handle key generation if part of a larger system.
    genome = neat.DefaultGenome(new_genome_key)
    genome.nodes = {}
    genome.connections = {}

    # Store original (serialized) IDs to map back hidden nodes if necessary for internal NEAT logic
    # However, the serialized format already uses remapped IDs for hidden nodes.
    # Output nodes use their direct IDs (0 to num_outputs - 1).
    # Hidden nodes are renumbered from num_outputs upwards.
    # So, the IDs parsed from the string are the final IDs for the new genome.

    # 3. Nodes Section
    current_token = next_token()
    if current_token == S_NODES_START:
        current_token = next_token() # Consume S_NODES_START, get first _NODE_ or S_CONNS_START or S_GEN_END
        while current_token == S_NODE_PREFIX: # S_NODE_PREFIX is "_NODE_"
            # Node: _NODE_ node_id bias response activation aggregation _E_NODES_
            try:
                node_id_str = next_token()
                bias_str = next_token()
                response_str = next_token()
                activation_str = next_token()
                aggregation_str = next_token()
                next_token(S_NODE_END_SUFFIX) # Consume _E_NODES_
            except ValueError as e: # Catches next_token errors or StopIteration
                raise ValueError(f"Malformed node definition: {e}")

            try:
                node_id = int(node_id_str)
                bias = float(bias_str)
                response = float(response_str)
            except ValueError as e:
                raise ValueError(f"Invalid node attribute value type for node {node_id_str}: {e}")

            if node_id in genome.nodes:
                raise ValueError(f"Duplicate node ID found in string: {node_id}")

            if not genome_config.activation_defs.is_valid(activation_str):
                raise ValueError(f"Invalid activation function '{activation_str}' for node {node_id} not in neat_config.")
            if not genome_config.aggregation_function_defs.is_valid(aggregation_str):
                raise ValueError(f"Invalid aggregation function '{aggregation_str}' for node {node_id} not in neat_config.")

            node_gene = genome_config.node_gene_type(node_id)
            # Manually set attributes instead of init_attributes, as they come from string
            node_gene.bias = bias
            node_gene.response = response
            node_gene.activation = activation_str
            node_gene.aggregation = aggregation_str
            genome.nodes[node_id] = node_gene

            current_token = next_token() # Get next _NODE_ or section start
    # If current_token was not S_NODES_START, it should be S_CONNS_START or S_GEN_END

    # 4. Connections Section
    if current_token == S_CONNS_START:
        current_token = next_token() # Consume S_CONNS_START, get first _CONN_ or S_GEN_END
        while current_token == S_CONN_PREFIX: # S_CONN_PREFIX is "_CONN_"
            # Connection: _CONN_ from_id to_id weight _E_CONNS_
            try:
                from_id_str = next_token()
                to_id_str = next_token()
                weight_str = next_token()
                next_token(S_CONN_END_SUFFIX) # Consume _E_CONNS_
            except ValueError as e: # Catches next_token errors or StopIteration
                raise ValueError(f"Malformed connection definition: {e}")

            try:
                from_id = int(from_id_str)
                to_id = int(to_id_str)
                weight = float(weight_str)
            except ValueError as e:
                raise ValueError(f"Invalid connection attribute value type for conn {from_id_str}->{to_id_str}: {e}")

            # Validate node IDs
            is_from_input_node = from_id < 0
            if is_from_input_node:
                # Ensure input ID is valid for the configured number of inputs
                # Input keys are -1, -2, ..., -num_inputs
                if not (-genome_config.num_inputs <= from_id <= -1):
                    raise ValueError(f"Connection 'from_node' ID {from_id} is an invalid input node ID for num_inputs={genome_config.num_inputs}.")
            elif from_id not in genome.nodes:
                raise ValueError(f"Connection 'from_node' ID {from_id} not defined in nodes section.")

            if to_id not in genome.nodes: # Output/Hidden nodes must be in genome.nodes
                raise ValueError(f"Connection 'to_node' ID {to_id} not defined in nodes section.")
            if to_id < 0 : # to_id cannot be an input node
                 raise ValueError(f"Connection 'to_node' ID {to_id} cannot be an input node.")


            conn_key = (from_id, to_id)
            if conn_key in genome.connections:
                raise ValueError(f"Duplicate connection key found in string: {conn_key}")

            conn_gene = genome_config.connection_gene_type(conn_key)
            # Manually set attributes
            conn_gene.weight = weight
            conn_gene.enabled = True # Connections in string are assumed enabled
            genome.connections[conn_key] = conn_gene

            current_token = next_token() # Get next _CONN_ or S_GEN_END
    # If current_token was not S_CONNS_START, it should be S_GEN_END

    # 5. Genome End
    if current_token != S_GEN_END:
        raise ValueError(f"Expected genome end '{S_GEN_END}', got '{current_token}'")

    # Check if there are any remaining tokens
    try:
        extra_token = next(token_iter, None)
        if extra_token is not None:
            raise ValueError(f"Extra tokens found after {S_GEN_END}. First extra: '{extra_token}'")
    except StopIteration: # This is expected if all tokens consumed
        pass

    genome.fitness = None # Fitness is not part of this serialization format

    # After populating nodes and connections, it might be necessary to
    # call genome.configure_new(neat_config) if the DefaultGenome expects this
    # to finalize some internal state, BUT we've manually constructed it.
    # The DefaultGenome.configure_new method typically *creates* initial nodes/connections.
    # Here, we are *reconstructing*, so direct assignment is more appropriate.
    # We need to ensure the genome's node_indexer is correctly set if new nodes are to be added later.
    # This can be done by calling config.get_new_node_key with the populated genome.nodes once.
    if genome.nodes:
        _ = genome_config.get_new_node_key(genome.nodes)


    return genome