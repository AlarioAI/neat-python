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
S_ATTR_SEP = "|"
# FLOAT_PRECISION = 6 # No longer used for fixed precision formatting

# --- Modified Float Formatting ---
def _format_float(value):
    """
    Helper to format floats to a string representation without unnecessary
    trailing zeros or decimal points.
    """
    try:
        # Format with sufficient precision, then strip trailing zeros and decimal points.
        # Using 10 decimal places during intermediate formatting handles most cases.
        s = "{:.10f}".format(value).rstrip('0').rstrip('.')
        # Ensure that a value like 0.0 becomes "0" and not "" or "."
        if s == '' or s == '-':
            return '0'
        return s
    except Exception:
        # Fallback to default string conversion if formatting fails
        return str(value)

# --- Serialize Genome Function (using the modified _format_float) ---
def serialize_genome(op_name: str, genome: neat.DefaultGenome, neat_config: neat.Config) -> str:
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
    lines.append(f"{S_GEN_START} </{op_name}>") # Use the format from the user spec

    # 2. Configuration Section
    conf_parts = [
        f"num_inputs{S_ATTR_SEP}{genome_config.num_inputs}",
        f"num_outputs{S_ATTR_SEP}{genome_config.num_outputs}",
        f"feed_forward{S_ATTR_SEP}{genome_config.feed_forward}",
        # Include gene types as requested in the format description
        f"node_gene_type{S_ATTR_SEP}{genome_config.node_gene_type.__name__}",
        f"connection_gene_type{S_ATTR_SEP}{genome_config.connection_gene_type.__name__}"
    ]
    lines.append(f"{S_CONF_START} {S_ATTR_SEP.join(conf_parts)} {S_CONF_END}")

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
                _format_float(node.bias),
                _format_float(node.response),
                node.activation,
                node.aggregation
            ])
            # Use the suffixes from the user spec
            all_nodes_serialized.append(f"{S_NODE_PREFIX}{node_str}{S_NODE_END_SUFFIX}")
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
            _format_float(node.bias),
            _format_float(node.response),
            node.activation,
            node.aggregation
        ])
        # Use the suffixes from the user spec
        all_nodes_serialized.append(f"{S_NODE_PREFIX}{node_str}{S_NODE_END_SUFFIX}")

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
            _format_float(conn.weight)
        ]
        # Use the suffixes from the user spec
        connections_serialized.append(
            (remapped_from_id, remapped_to_id, f"{S_CONN_PREFIX}{S_ATTR_SEP.join(conn_str_parts)}{S_CONN_END_SUFFIX}")
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

    return "\n".join(lines)


# --- Deserialize Genome Function (Does not need modification for this request) ---
def deserialize_genome(genome_string: str, neat_config: neat.Config) -> neat.DefaultGenome:
    """
    Deserializes a string representation back into a NEAT genome.
    (Code is the same as in neat_context.txt/previous examples,
     as it parses the string representation created by serialize_genome.
     No changes needed here to handle the absence of trailing zeros,
     as float() handles various valid float string formats.)

    Args:
        genome_string: The string representation of the genome.
        neat_config: The neat.Config object to use for creating gene instances
                     and providing context. It's crucial that this config is
                     compatible with the serialized genome's parameters
                     (gene types, activation/aggregation functions available).
    Returns:
        A neat.DefaultGenome object.
    Raises:
        ValueError: If the string format is invalid or inconsistent.
    """
    lines = genome_string.strip().split('\n')
    line_idx = 0
    parsed_conf = {} # Store parsed config values from string

    # --- Helper to get next line or raise error ---
    def get_next_line():
        nonlocal line_idx
        if line_idx >= len(lines):
            raise ValueError("Unexpected end of string: More lines expected.")
        line = lines[line_idx]
        line_idx += 1
        return line.strip()

    # 1. Genome Start and Operation Name
    line = get_next_line()
    if not line.startswith(S_GEN_START + " </"):
        raise ValueError(f"Expected genome start with op_name (e.g., '{S_GEN_START} </op>'), got '{line}'")
    # op_name can be extracted here if needed by the caller, but deserialize doesn't use it internally

    # 2. Configuration Section
    line = get_next_line()
    if not line.startswith(S_CONF_START) or not line.endswith(S_CONF_END):
        raise ValueError(f"Malformed config line: '{line}' expected start '{S_CONF_START}' and end '{S_CONF_END}'")
    conf_content = line[len(S_CONF_START):-len(S_CONF_END)].strip()
    conf_parts = conf_content.split(S_ATTR_SEP)

    # Ensure non-empty and even number of parts for key-value pairs
    if not conf_parts or len(conf_parts) % 2 != 0:
         # Handle specific case like S_CONF_START + S_ATTR_SEP + ... which results in leading empty string
        if len(conf_parts) > 0 and conf_parts[0] == '':
            conf_parts = conf_parts[1:] # Adjust if split resulted in leading empty string
            if len(conf_parts) % 2 != 0: # Re-check after adjustment
                raise ValueError(f"Malformed config content (key-value pairs expected): '{conf_content}'")
        else:
            raise ValueError(f"Malformed config content (key-value pairs expected): '{conf_content}'")


    for i in range(0, len(conf_parts), 2):
        key = conf_parts[i].strip()
        value = conf_parts[i+1].strip()
        if not key: # Avoid empty keys if separators are adjacent or at ends
            raise ValueError(f"Empty key found in config content: '{conf_content}'")
        parsed_conf[key] = value

    # Validate essential config against provided neat_config
    genome_config = neat_config.genome_config
    try:
        s_num_inputs = int(parsed_conf['num_inputs'])
        s_num_outputs = int(parsed_conf['num_outputs'])
        s_feed_forward = parsed_conf['feed_forward'] == "True" # Case sensitive comparison
        s_node_gene_type_name = parsed_conf['node_gene_type']
        s_conn_gene_type_name = parsed_conf['connection_gene_type']

        if s_num_inputs != genome_config.num_inputs:
            raise ValueError(f"Config mismatch: num_inputs in string ({s_num_inputs}) != neat_config ({genome_config.num_inputs})")
        if s_num_outputs != genome_config.num_outputs:
            raise ValueError(f"Config mismatch: num_outputs in string ({s_num_outputs}) != neat_config ({genome_config.num_outputs})")
        if s_feed_forward != genome_config.feed_forward:
            raise ValueError(f"Config mismatch: feed_forward in string ({s_feed_forward}) != neat_config ({genome_config.feed_forward})")
        if s_node_gene_type_name != genome_config.node_gene_type.__name__:
            raise ValueError(f"Config mismatch: node_gene_type in string ('{s_node_gene_type_name}') != neat_config ('{genome_config.node_gene_type.__name__}')")
        if s_conn_gene_type_name != genome_config.connection_gene_type.__name__:
            raise ValueError(f"Config mismatch: connection_gene_type in string ('{s_conn_gene_type_name}') != neat_config ('{genome_config.connection_gene_type.__name__}')")

    except KeyError as e:
        raise ValueError(f"Missing essential key in config string: {e}. Found keys: {list(parsed_conf.keys())}")
    except ValueError as e:
        raise ValueError(f"Invalid value type or mismatch in config string: {e}")


    # Create a new genome. Key is not stored, assign a default (e.g., 0)
    new_genome_key = 0 # Placeholder key
    genome = neat.DefaultGenome(new_genome_key)
    genome.nodes = {}
    genome.connections = {}

    # 3. Nodes Section
    expected_node_suffix = S_NODE_END_SUFFIX # From user spec
    if line_idx < len(lines) and lines[line_idx].strip() == S_NODES_START:
        get_next_line() # Consume S_NODES_START
        while line_idx < len(lines) and lines[line_idx].strip().startswith(S_NODE_PREFIX):
            line = get_next_line()
            if not line.endswith(expected_node_suffix):
                 raise ValueError(f"Malformed node line: '{line}'. Expected suffix '{expected_node_suffix}'")
            node_content = line[len(S_NODE_PREFIX):-len(expected_node_suffix)].strip()
            node_attrs = node_content.split(S_ATTR_SEP)
            if len(node_attrs) != 5: # id|bias|response|activation|aggregation
                raise ValueError(f"Malformed node attributes: '{node_content}' (expected 5 parts separated by '{S_ATTR_SEP}')")

            try:
                node_id = int(node_attrs[0])
                bias = float(node_attrs[1]) # float() handles the compact string format
                response = float(node_attrs[2]) # float() handles the compact string format
                activation = node_attrs[3]
                aggregation = node_attrs[4]
            except ValueError as e:
                 raise ValueError(f"Invalid node attribute value type: {e} in '{node_content}'")

            if node_id in genome.nodes:
                raise ValueError(f"Duplicate node ID found in string: {node_id}")

            # Ensure activation/aggregation functions are valid for the given config
            if not genome_config.activation_defs.is_valid(activation):
                 raise ValueError(f"Invalid activation function '{activation}' not defined in neat_config.")
            if not genome_config.aggregation_function_defs.is_valid(aggregation):
                raise ValueError(f"Invalid aggregation function '{aggregation}' not defined in neat_config.")

            # Create node gene and set attributes directly
            node_gene = genome_config.node_gene_type(node_id)
            node_gene.bias = bias
            node_gene.response = response
            node_gene.activation = activation
            node_gene.aggregation = aggregation
            genome.nodes[node_id] = node_gene
        # Check if the next line is S_CONNS_START or S_GEN_END
        if line_idx < len(lines) and lines[line_idx].strip() not in [S_CONNS_START, S_GEN_END]:
            # If S_NODES_START was present, but parsing stopped unexpectedly
             raise ValueError(f"Expected end of nodes or start of connections, but found: '{lines[line_idx].strip()}'")

    # 4. Connections Section
    expected_conn_suffix = S_CONN_END_SUFFIX # From user spec
    if line_idx < len(lines) and lines[line_idx].strip() == S_CONNS_START:
        get_next_line() # Consume S_CONNS_START
        while line_idx < len(lines) and lines[line_idx].strip().startswith(S_CONN_PREFIX):
            line = get_next_line()
            if not line.endswith(expected_conn_suffix):
                raise ValueError(f"Malformed connection line: '{line}'. Expected suffix '{expected_conn_suffix}'")
            conn_content = line[len(S_CONN_PREFIX):-len(expected_conn_suffix)].strip()
            conn_attrs = conn_content.split(S_ATTR_SEP)
            if len(conn_attrs) != 3: # from_id|to_id|weight
                 raise ValueError(f"Malformed connection attributes: '{conn_content}' (expected 3 parts separated by '{S_ATTR_SEP}')")

            try:
                from_id = int(conn_attrs[0])
                to_id = int(conn_attrs[1])
                weight = float(conn_attrs[2]) # float() handles the compact string format
            except ValueError as e:
                 raise ValueError(f"Invalid connection attribute value type: {e} in '{conn_content}'")

            # Validate node IDs used in connections
            # from_id can be an input node (<0) or an existing (output/remapped hidden) node.
            is_from_input = from_id < 0
            if is_from_input:
                 # Check if input ID is valid based on num_inputs
                 if (s_num_inputs + from_id < 0): # e.g., num_inputs=2, valid inputs are -1, -2. If from_id=-3, 2+(-3)<0.
                     raise ValueError(f"Connection 'from_node' ID {from_id} is an invalid input node ID for num_inputs={s_num_inputs}.")
            elif from_id not in genome.nodes:
                 # Check if the 'from' node (which is not an input) exists in the parsed nodes
                 raise ValueError(f"Connection 'from_node' ID {from_id} not defined in nodes section.")

            # to_id must be an existing (output/remapped hidden) node. It cannot be an input node.
            if to_id < 0:
                raise ValueError(f"Connection 'to_node' ID {to_id} cannot be an input node.")
            if to_id not in genome.nodes:
                 raise ValueError(f"Connection 'to_node' ID {to_id} not defined in nodes section.")


            conn_key = (from_id, to_id)
            if conn_key in genome.connections:
                raise ValueError(f"Duplicate connection key found in string: {conn_key}")

            # Create connection gene and set attributes
            conn_gene = genome_config.connection_gene_type(conn_key)
            conn_gene.weight = weight
            conn_gene.enabled = True # Only enabled connections are serialized
            genome.connections[conn_key] = conn_gene
        # Check if the next line is S_GEN_END
        if line_idx < len(lines) and lines[line_idx].strip() != S_GEN_END:
             # If S_CONNS_START was present, but parsing stopped unexpectedly
            raise ValueError(f"Expected end of connections or genome end, but found: '{lines[line_idx].strip()}'")

    # 5. Genome End
    line = get_next_line()
    if line != S_GEN_END:
        raise ValueError(f"Expected genome end '{S_GEN_END}', got '{line}'")

    # Check for unexpected extra lines
    if line_idx < len(lines):
        raise ValueError(f"Extra lines found after genome end marker '{S_GEN_END}'. First extra: '{lines[line_idx]}'")

    # Fitness is not part of serialization, default to None.
    genome.fitness = None
    return genome