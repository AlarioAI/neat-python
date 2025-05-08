import unittest
import os
import neat # Assuming neat.genome, neat.config, neat.genes are accessible
from genome_srl import serialize_genome, deserialize_genome # Assuming functions are in genome_srl.py

# --- Constants from genome_srl.py for direct use in tests if needed ---
S_GEN_START = "_S_GEN_"
S_GEN_END = "_E_GEN_"
S_CONF_START = "_S_CONF_"
S_CONF_END = "_E_CONF_"
S_NODES_START = "_S_NODES_"
S_NODE_PREFIX = "_NODE_"
S_NODE_END_SUFFIX = "_E_NODES_"
S_CONNS_START = "_S_CONNS_"
S_CONN_PREFIX = "_CONN_"
S_CONN_END_SUFFIX = "_E_CONNS_"
S_ATTR_SEP = "|"
FLOAT_PRECISION = 6

def _format_float(value):
    return f"{value:.{FLOAT_PRECISION}f}"

class TestGenomeSerialization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up a NEAT config object that can be used across tests.
        This attempts to load 'xor/config-feedforward'.
        A minimal config could also be constructed manually if preferred.
        """
        # Determine path to configuration file.
        # This assumes tests are run from a directory where 'examples/xor/config-feedforward' is accessible.
        # Or that the test file is in 'examples/' and config is in 'examples/xor/'
        cls.config = None
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Try path relative to current test file location
            # Assumes test_genome_srl.py might be in `examples` or `examples/some_subdir`
            # and config is in `examples/xor/`

            # Path if test file is in `examples`
            config_path_1 = os.path.join(current_dir, "xor", "config-feedforward")
            # Path if test file is in a subdir of `examples` (like `examples/tests`)
            config_path_2 = os.path.join(os.path.dirname(current_dir), "xor", "config-feedforward")
            # Path if test file is in `examples/xor`
            config_path_3 = os.path.join(current_dir, "config-feedforward")


            if os.path.exists(config_path_1):
                config_path = config_path_1
            elif os.path.exists(config_path_2):
                config_path = config_path_2
            elif os.path.exists(config_path_3):
                config_path = config_path_3
            else:
                # Fallback for contexts where the above might not work (e.g. running from root)
                # This is a common location if 'neat-python' is cloned.
                module_root = os.path.join(current_dir, "..") # Go up one level from where test_genome_srl.py might be
                config_path_root_relative = os.path.join(module_root, "examples", "xor", "config-feedforward")
                if os.path.exists(config_path_root_relative):
                    config_path = config_path_root_relative
                else:
                    raise FileNotFoundError(f"Config file not found. Tried: {config_path_1}, {config_path_2}, {config_path_3}, {config_path_root_relative}")

            cls.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_path)
        except Exception as e:
            print(f"Could not load NEAT config for tests: {e}. Some tests might fail or be inaccurate.")
            # Create a minimal mock config if loading fails, to allow some tests to run
            cls.config = cls.create_minimal_mock_config()

    @staticmethod
    def create_minimal_mock_config():
        """Creates a very basic mock NEAT config for testing if file loading fails."""
        class MockGenomeConfig:
            num_inputs = 2
            num_outputs = 1
            input_keys = [-1, -2]
            output_keys = [0]
            feed_forward = True
            node_gene_type = neat.genes.DefaultNodeGene
            connection_gene_type = neat.genes.DefaultConnectionGene
            # Mock activation/aggregation defs
            activation_defs = neat.activations.ActivationFunctionSet()
            aggregation_function_defs = neat.aggregations.AggregationFunctionSet()

            # Attributes for DefaultNodeGene and DefaultConnectionGene
            bias_init_mean, bias_init_stdev, bias_replace_rate, bias_mutate_rate, bias_mutate_power, bias_max_value, bias_min_value = 0.0,1.0,0.1,0.7,0.5,5.0,-5.0
            response_init_mean, response_init_stdev, response_replace_rate, response_mutate_rate, response_mutate_power, response_max_value, response_min_value = 1.0,0.1,0.1,0.1,0.1,5.0,-5.0
            activation_default, activation_options, activation_mutate_rate = "sigmoid", ["sigmoid", "relu", "identity"], 0.1
            aggregation_default, aggregation_options, aggregation_mutate_rate = "sum", ["sum"], 0.1
            weight_init_mean, weight_init_stdev, weight_replace_rate, weight_mutate_rate, weight_mutate_power, weight_max_value, weight_min_value = 0.0,1.0,0.1,0.8,0.5,5.0,-5.0
            enabled_default, enabled_mutate_rate = "True", 0.01


        class MockConfig:
            genome_config = MockGenomeConfig()
            # Add other top-level config sections if needed by DefaultGenome/DefaultReproduction etc.
            num_inputs = 2 # For compatibility if DefaultGenome.configure_new is called
            num_outputs = 1
            num_hidden = 0 # For configure_new
            initial_connection = "unconnected" # For configure_new

            # Mock methods/attributes of neat.Config if they are directly accessed by tested functions
            # For serialize/deserialize, genome_config is the main part.

        return MockConfig()


    def _create_sample_genome(self, key=1, num_hidden=1, add_disabled_conn=False):
        """Helper to create a genome for testing."""
        genome = neat.DefaultGenome(key)
        # Use configure_new to set up inputs/outputs based on config
        # For this to work, self.config needs to be a full neat.Config object
        # or a mock that DefaultGenome.configure_new can use.

        # Manually adjust num_hidden in a copy of the config for this specific genome if needed
        # This is tricky because configure_new uses the config passed to it.
        # For robust testing, it's better if cls.config is a real loaded config.
        # If using mock, ensure num_hidden is set there.

        # For simplicity in this helper, let's manually define nodes and connections
        # This bypasses configure_new's direct use of config.num_hidden for this setup

        genome_config = self.config.genome_config

        # Output node
        genome.nodes[0] = genome_config.node_gene_type(0)
        genome.nodes[0].init_attributes(genome_config)
        genome.nodes[0].bias = 0.1
        genome.nodes[0].response = 1.1
        genome.nodes[0].activation = "sigmoid"
        genome.nodes[0].aggregation = "sum"

        if num_hidden > 0:
            # Hidden node ID: output_keys_max + 1 + i
            # Max output ID is num_outputs - 1. So hidden starts at num_outputs.
            hidden_id_start = genome_config.num_outputs
            for i in range(num_hidden):
                hid = hidden_id_start + i
                genome.nodes[hid] = genome_config.node_gene_type(hid)
                genome.nodes[hid].init_attributes(genome_config)
                genome.nodes[hid].bias = 0.2 + i * 0.1
                genome.nodes[hid].response = 1.2 + i * 0.1
                genome.nodes[hid].activation = "relu"
                genome.nodes[hid].aggregation = "sum"

        # Connections
        # Input -1 to Output 0
        conn1_key = (-1, 0)
        genome.connections[conn1_key] = genome_config.connection_gene_type(conn1_key)
        genome.connections[conn1_key].init_attributes(genome_config)
        genome.connections[conn1_key].weight = 0.5
        genome.connections[conn1_key].enabled = True

        if num_hidden > 0:
            hidden_id_0 = genome_config.num_outputs # First hidden node
            # Input -2 to Hidden hidden_id_0
            conn2_key = (-2, hidden_id_0)
            genome.connections[conn2_key] = genome_config.connection_gene_type(conn2_key)
            genome.connections[conn2_key].init_attributes(genome_config)
            genome.connections[conn2_key].weight = -0.3
            genome.connections[conn2_key].enabled = True

            # Hidden hidden_id_0 to Output 0
            conn3_key = (hidden_id_0, 0)
            genome.connections[conn3_key] = genome_config.connection_gene_type(conn3_key)
            genome.connections[conn3_key].init_attributes(genome_config)
            genome.connections[conn3_key].weight = 0.7
            genome.connections[conn3_key].enabled = True

            if num_hidden > 1: # Add one more connection for a second hidden node
                hidden_id_1 = genome_config.num_outputs + 1
                conn4_key = (hidden_id_0, hidden_id_1) # Hidden to Hidden
                genome.connections[conn4_key] = genome_config.connection_gene_type(conn4_key)
                genome.connections[conn4_key].init_attributes(genome_config)
                genome.connections[conn4_key].weight = 0.4
                genome.connections[conn4_key].enabled = True


        if add_disabled_conn:
            # Input -1 to Hidden (if exists) or another connection
            target_for_disabled = genome_config.num_outputs if num_hidden > 0 else 0
            # Ensure target_for_disabled is not an input node if it's 0 (output)
            if target_for_disabled < 0: target_for_disabled = 0

            # Create a unique key for the disabled connection
            # Try connecting input -1 to the first hidden/output node
            # If that's already conn1_key, try input -2.
            disabled_conn_key = (-1, target_for_disabled)
            if disabled_conn_key == conn1_key: # If (-1,0) is already used
                 disabled_conn_key = (-2, target_for_disabled) # Try (-2,0) or (-2,hidden_id_0)
                 # Ensure this new key is also not already taken if target is hidden_id_0
                 if num_hidden > 0 and disabled_conn_key == (-2, genome_config.num_outputs): # conn2_key
                     # This case is getting complex; for a simple disabled connection,
                     # let's just ensure it's a new one.
                     # If hidden node exists, connect input -1 to it if not already.
                     if num_hidden > 0:
                        disabled_conn_key_alt = (-1, genome_config.num_outputs)
                        if disabled_conn_key_alt not in genome.connections:
                             disabled_conn_key = disabled_conn_key_alt

            if disabled_conn_key not in genome.connections:
                genome.connections[disabled_conn_key] = genome_config.connection_gene_type(disabled_conn_key)
                genome.connections[disabled_conn_key].init_attributes(genome_config)
                genome.connections[disabled_conn_key].weight = 0.99
                genome.connections[disabled_conn_key].enabled = False
            elif (-2,0) not in genome.connections: # Fallback if others were taken
                disabled_conn_key = (-2,0)
                genome.connections[disabled_conn_key] = genome_config.connection_gene_type(disabled_conn_key)
                genome.connections[disabled_conn_key].init_attributes(genome_config)
                genome.connections[disabled_conn_key].weight = 0.99
                genome.connections[disabled_conn_key].enabled = False


        return genome

    def assertGenomesEffectivelyEqual(self, g1_orig, g2_deserialized, neat_config, num_hidden_orig):
        """
        Compares two genomes, accounting for hidden node renumbering and
        that g2_deserialized only contains enabled connections.
        """
        genome_config = neat_config.genome_config
        num_outputs = genome_config.num_outputs

        # 1. Compare Config (implicitly checked by deserialize_genome)

        # 2. Compare Nodes
        # Output nodes (IDs 0 to num_outputs-1)
        for i in range(num_outputs):
            self.assertIn(i, g1_orig.nodes)
            self.assertIn(i, g2_deserialized.nodes)
            n1 = g1_orig.nodes[i]
            n2 = g2_deserialized.nodes[i]
            self.assertEqual(n1.bias, n2.bias, msg=f"Bias mismatch for output node {i}")
            self.assertEqual(n1.response, n2.response, msg=f"Response mismatch for output node {i}")
            self.assertEqual(n1.activation, n2.activation, msg=f"Activation mismatch for output node {i}")
            self.assertEqual(n1.aggregation, n2.aggregation, msg=f"Aggregation mismatch for output node {i}")

        # Hidden nodes (original IDs vs remapped IDs)
        original_hidden_ids = sorted([nid for nid in g1_orig.nodes if nid >= num_outputs]) # Assuming original hidden IDs also start after outputs

        # If the _create_sample_genome uses original hidden IDs starting from num_outputs, this works.
        # If original hidden IDs are arbitrary large numbers from NEAT evolution, this needs adjustment.
        # For this test, _create_sample_genome creates hidden nodes with IDs starting from num_outputs.
        # So, original_hidden_ids from g1_orig will be [num_outputs, num_outputs+1, ...]
        # The remapped IDs in g2_deserialized for these will be the same.

        self.assertEqual(len(original_hidden_ids), num_hidden_orig, "Number of hidden nodes mismatch in original genome setup")

        num_deserialized_hidden = len([nid for nid in g2_deserialized.nodes if nid >= num_outputs])
        self.assertEqual(num_deserialized_hidden, num_hidden_orig, "Number of deserialized hidden nodes mismatch")


        for i in range(num_hidden_orig):
            orig_hid = original_hidden_ids[i] # Original ID from g1_orig
            remapped_hid = num_outputs + i    # Expected remapped ID in g2_deserialized

            self.assertIn(orig_hid, g1_orig.nodes)
            self.assertIn(remapped_hid, g2_deserialized.nodes)
            n1 = g1_orig.nodes[orig_hid]
            n2 = g2_deserialized.nodes[remapped_hid]
            self.assertAlmostEqual(n1.bias, n2.bias, places=FLOAT_PRECISION, msg=f"Bias mismatch for hidden node {orig_hid}->{remapped_hid}")
            self.assertAlmostEqual(n1.response, n2.response, places=FLOAT_PRECISION, msg=f"Response mismatch for hidden node {orig_hid}->{remapped_hid}")
            self.assertEqual(n1.activation, n2.activation, msg=f"Activation mismatch for hidden node {orig_hid}->{remapped_hid}")
            self.assertEqual(n1.aggregation, n2.aggregation, msg=f"Aggregation mismatch for hidden node {orig_hid}->{remapped_hid}")


        # 3. Compare Connections (only enabled ones from g1_orig)
        g1_enabled_conn_keys_remapped = set()
        g1_enabled_conns_remapped_details = {}

        hidden_id_map_test = {orig_id: num_outputs + i for i, orig_id in enumerate(original_hidden_ids)}

        for (from_orig, to_orig), conn1 in g1_orig.connections.items():
            if conn1.enabled:
                # Remap from_orig
                if from_orig < 0: remapped_from = from_orig
                elif from_orig < num_outputs: remapped_from = from_orig
                else: remapped_from = hidden_id_map_test.get(from_orig)

                # Remap to_orig
                if to_orig < num_outputs: remapped_to = to_orig
                else: remapped_to = hidden_id_map_test.get(to_orig)

                if remapped_from is None or remapped_to is None: continue # Should not happen with consistent test data

                remapped_key = (remapped_from, remapped_to)
                g1_enabled_conn_keys_remapped.add(remapped_key)
                g1_enabled_conns_remapped_details[remapped_key] = conn1.weight

        g2_conn_keys = set(g2_deserialized.connections.keys())
        self.assertEqual(g1_enabled_conn_keys_remapped, g2_conn_keys, "Remapped enabled connection keys mismatch")

        for key2, conn2 in g2_deserialized.connections.items():
            self.assertTrue(conn2.enabled) # All deserialized conns should be enabled
            self.assertIn(key2, g1_enabled_conns_remapped_details, f"Connection {key2} in deserialized but not in remapped original.")
            self.assertAlmostEqual(g1_enabled_conns_remapped_details[key2], conn2.weight, places=FLOAT_PRECISION,
                                   msg=f"Weight mismatch for connection {key2}")


    def test_happy_path_simple_ff(self):
        """Test serialization and deserialization of a simple feed-forward genome."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        op_name = "test_xor"
        num_hidden = 1
        genome_orig = self._create_sample_genome(key=1, num_hidden=num_hidden, add_disabled_conn=True)

        serialized_str = serialize_genome(op_name, genome_orig, self.config)
        self.assertIsInstance(serialized_str, str)
        # print("\n--- Serialized Happy Path ---")
        # print(serialized_str)
        # print("--- End Serialized ---")


        genome_deserialized = deserialize_genome(serialized_str, self.config)
        self.assertIsInstance(genome_deserialized, neat.DefaultGenome)

        self.assertGenomesEffectivelyEqual(genome_orig, genome_deserialized, self.config, num_hidden)

        # Test network creation
        try:
            net = neat.nn.FeedForwardNetwork.create(genome_deserialized, self.config)
            self.assertIsNotNone(net)
            # inputs = [0.0] * self.config.genome_config.num_inputs
            # output = net.activate(inputs) # This might fail if inputs don't match
            # self.assertIsNotNone(output)
        except Exception as e:
            self.fail(f"FeedForwardNetwork creation failed for deserialized genome: {e}\nSerialized string:\n{serialized_str}")


    def test_edge_case_no_hidden_nodes(self):
        """Test with a genome that has no hidden nodes."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        op_name = "no_hidden"
        num_hidden = 0
        genome_orig = self._create_sample_genome(key=2, num_hidden=num_hidden)
        # Ensure no hidden nodes for this test case in the created sample
        genome_orig.nodes = {k:v for k,v in genome_orig.nodes.items() if k < self.config.genome_config.num_outputs} # Keep only output
        # Remove connections involving non-existent hidden nodes
        genome_orig.connections = {k:v for k,v in genome_orig.connections.items() if k[0] < 0 and k[1] < self.config.genome_config.num_outputs}


        serialized_str = serialize_genome(op_name, genome_orig, self.config)
        # print("\n--- Serialized No Hidden ---")
        # print(serialized_str)
        # print("--- End Serialized ---")
        self.assertNotIn(S_NODE_PREFIX + str(self.config.genome_config.num_outputs), serialized_str) # Check no remapped hidden node ID

        genome_deserialized = deserialize_genome(serialized_str, self.config)
        self.assertGenomesEffectivelyEqual(genome_orig, genome_deserialized, self.config, num_hidden)


    def test_edge_case_no_enabled_connections(self):
        """Test with a genome that has no enabled connections."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        op_name = "no_conns"
        num_hidden = 1
        genome_orig = self._create_sample_genome(key=3, num_hidden=num_hidden)
        for conn in genome_orig.connections.values():
            conn.enabled = False

        serialized_str = serialize_genome(op_name, genome_orig, self.config)
        # print("\n--- Serialized No Enabled Conns ---")
        # print(serialized_str)
        # print("--- End Serialized ---")
        self.assertNotIn(S_CONNS_START, serialized_str) # No connections section should be present

        genome_deserialized = deserialize_genome(serialized_str, self.config)
        self.assertGenomesEffectivelyEqual(genome_orig, genome_deserialized, self.config, num_hidden)
        self.assertEqual(len(genome_deserialized.connections), 0)


    def test_error_malformed_string_missing_marker(self):
        """Test deserialization with a string missing a critical marker."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        bad_string = f"{S_GEN_START} </test>\n...missing config...\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Malformed config line"):
            deserialize_genome(bad_string, self.config)

    def test_error_malformed_string_bad_conf_values(self):
        """Test deserialization with non-numeric values in config where numbers expected."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        conf_line = f"{S_CONF_START} num_inputs{S_ATTR_SEP}TWO{S_ATTR_SEP}num_outputs{S_ATTR_SEP}1{S_ATTR_SEP}feed_forward{S_ATTR_SEP}True{S_ATTR_SEP}node_gene_type{S_ATTR_SEP}DefaultNodeGene{S_ATTR_SEP}connection_gene_type{S_ATTR_SEP}DefaultConnectionGene {S_CONF_END}"
        bad_string = f"{S_GEN_START} </test>\n{conf_line}\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Invalid value type in config string: invalid literal for int\(\) with base 10: 'TWO'"):
            deserialize_genome(bad_string, self.config)

    def test_error_config_mismatch(self):
        """Test deserialization when string's config mismatches neat_config."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        # Create a config string with num_inputs=3, but our self.config has num_inputs=2
        conf_line = f"{S_CONF_START} num_inputs{S_ATTR_SEP}3{S_ATTR_SEP}num_outputs{S_ATTR_SEP}1{S_ATTR_SEP}feed_forward{S_ATTR_SEP}True{S_ATTR_SEP}node_gene_type{S_ATTR_SEP}DefaultNodeGene{S_ATTR_SEP}connection_gene_type{S_ATTR_SEP}DefaultConnectionGene {S_CONF_END}"
        mismatch_string = f"{S_GEN_START} </test>\n{conf_line}\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Config mismatch: num_inputs in string \(3\) != neat_config \(2\)"):
            deserialize_genome(mismatch_string, self.config)

    def test_error_node_attribute_count(self):
        """Test node line with wrong number of attributes."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        conf_line = f"{S_CONF_START} num_inputs{S_ATTR_SEP}2{S_ATTR_SEP}num_outputs{S_ATTR_SEP}1{S_ATTR_SEP}feed_forward{S_ATTR_SEP}True{S_ATTR_SEP}node_gene_type{S_ATTR_SEP}DefaultNodeGene{S_ATTR_SEP}connection_gene_type{S_ATTR_SEP}DefaultConnectionGene {S_CONF_END}"
        node_line = f"{S_NODE_PREFIX}0{S_ATTR_SEP}0.1{S_ATTR_SEP}1.1{S_ATTR_SEP}sigmoid {S_NODE_END_SUFFIX}" # Missing aggregation
        bad_string = f"{S_GEN_START} </test>\n{conf_line}\n{S_NODES_START}\n{node_line}\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Malformed node attributes.*expected 5 parts"):
            deserialize_genome(bad_string, self.config)

    def test_error_conn_attribute_count(self):
        """Test connection line with wrong number of attributes."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        conf_line = f"{S_CONF_START} num_inputs{S_ATTR_SEP}2{S_ATTR_SEP}num_outputs{S_ATTR_SEP}1{S_ATTR_SEP}feed_forward{S_ATTR_SEP}True{S_ATTR_SEP}node_gene_type{S_ATTR_SEP}DefaultNodeGene{S_ATTR_SEP}connection_gene_type{S_ATTR_SEP}DefaultConnectionGene {S_CONF_END}"
        # Output node 0
        node0_line = f"{S_NODE_PREFIX}0{S_ATTR_SEP}{_format_float(0.1)}{S_ATTR_SEP}{_format_float(1.1)}{S_ATTR_SEP}sigmoid{S_ATTR_SEP}sum {S_NODE_END_SUFFIX}"
        conn_line = f"{S_CONN_PREFIX}-1{S_ATTR_SEP}0 {S_CONN_END_SUFFIX}" # Missing weight
        bad_string = f"{S_GEN_START} </test>\n{conf_line}\n{S_NODES_START}\n{node0_line}\n{S_CONNS_START}\n{conn_line}\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Malformed connection attributes.*expected 3 parts"):
            deserialize_genome(bad_string, self.config)

    def test_error_invalid_activation_func(self):
        """Test deserialization with an activation function not in neat_config."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        conf_line = f"{S_CONF_START} num_inputs{S_ATTR_SEP}2{S_ATTR_SEP}num_outputs{S_ATTR_SEP}1{S_ATTR_SEP}feed_forward{S_ATTR_SEP}True{S_ATTR_SEP}node_gene_type{S_ATTR_SEP}DefaultNodeGene{S_ATTR_SEP}connection_gene_type{S_ATTR_SEP}DefaultConnectionGene {S_CONF_END}"
        node_line = f"{S_NODE_PREFIX}0{S_ATTR_SEP}{_format_float(0.1)}{S_ATTR_SEP}{_format_float(1.1)}{S_ATTR_SEP}non_existent_activation{S_ATTR_SEP}sum {S_NODE_END_SUFFIX}"
        bad_string = f"{S_GEN_START} </test>\n{conf_line}\n{S_NODES_START}\n{node_line}\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Invalid activation function 'non_existent_activation' not in neat_config."):
            deserialize_genome(bad_string, self.config)

    def test_error_connection_to_undefined_node(self):
        """Test connection to a node ID not defined in nodes section or as input."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        conf_line = f"{S_CONF_START} num_inputs{S_ATTR_SEP}2{S_ATTR_SEP}num_outputs{S_ATTR_SEP}1{S_ATTR_SEP}feed_forward{S_ATTR_SEP}True{S_ATTR_SEP}node_gene_type{S_ATTR_SEP}DefaultNodeGene{S_ATTR_SEP}connection_gene_type{S_ATTR_SEP}DefaultConnectionGene {S_CONF_END}"
        # Only output node 0 is defined
        node0_line = f"{S_NODE_PREFIX}0{S_ATTR_SEP}{_format_float(0.1)}{S_ATTR_SEP}{_format_float(1.1)}{S_ATTR_SEP}sigmoid{S_ATTR_SEP}sum {S_NODE_END_SUFFIX}"
        # Connection from input -1 to undefined node 1 (assuming only output 0 exists, so 1 is not a valid remapped hidden)
        conn_line = f"{S_CONN_PREFIX}-1{S_ATTR_SEP}1{S_ATTR_SEP}{_format_float(0.5)} {S_CONN_END_SUFFIX}"
        bad_string = f"{S_GEN_START} </test>\n{conf_line}\n{S_NODES_START}\n{node0_line}\n{S_CONNS_START}\n{conn_line}\n{S_GEN_END}"
        with self.assertRaisesRegex(ValueError, "Connection 'to_node' ID 1 not defined in nodes section."):
            deserialize_genome(bad_string, self.config)

    def test_precision_floats(self):
        """Test that float precision is handled as expected."""
        if not self.config: self.skipTest("NEAT Config not loaded.")
        op_name = "precision_test"
        genome = self._create_sample_genome(key=10)
        genome.nodes[0].bias = 1/3  # 0.333333...
        genome.connections[(-1,0)].weight = 2/3 # 0.666666...

        serialized = serialize_genome(op_name, genome, self.config)

        # Expected string for node 0 bias
        expected_bias_str = _format_float(1/3) # e.g., "0.333333"
        self.assertIn(f"{S_NODE_PREFIX}0{S_ATTR_SEP}{expected_bias_str}", serialized)

        # Expected string for connection (-1,0) weight
        expected_weight_str = _format_float(2/3) # e.g., "0.666667" (due to rounding if precision is 6)
        self.assertIn(f"{S_CONN_PREFIX}-1{S_ATTR_SEP}0{S_ATTR_SEP}{expected_weight_str}", serialized)

        deserialized = deserialize_genome(serialized, self.config)
        self.assertAlmostEqual(deserialized.nodes[0].bias, float(expected_bias_str), places=FLOAT_PRECISION)
        self.assertAlmostEqual(deserialized.connections[(-1,0)].weight, float(expected_weight_str), places=FLOAT_PRECISION)


if __name__ == '__main__':
    unittest.main()
