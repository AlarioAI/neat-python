import inspect
from collections import deque
from typing import Callable

import graphviz

from neat.nn.ops import sigmoid
from neat.nn.ops import tanh
from neat.nn.ops import relu
from neat.nn.ops import identity

class Node:
    def __init__(self, label, activation:Callable, state:float=0., bias:float=0):
        self._label = label
        self._state = state
        self._bias = bias
        self._pre_fire:float = 0
        self._activation = activation
        self._validate_activation()
        self._outgoing_edges = []

    def _validate_activation(self):
        if not callable(self._activation):
            raise ValueError("activation must be a callable function")

        # Get the signature of the function
        sig = inspect.signature(self._activation)

        # Check the number of parameters
        if len(sig.parameters) != 1:
            raise ValueError("activation must be a unary function")

        # Check if the function can handle floats
        try:
            result = self._activation(1.0)
            if not isinstance(result, float):
                raise ValueError("activation must return a float")
        except Exception as exc:
            raise ValueError(f'activation must be able to handle float inputs\n{exc}') from exc

    @property
    def label(self):
        return self._label

    @property
    def state(self):
        return self._state
    
    @property
    def bias(self):
        return self._bias

    @state.setter
    def state(self, value):
        self._state = value

    @bias.setter
    def bias(self, value):
        self._bias = value
        
    def add_edge(self, edge):
        self._outgoing_edges.append(edge)

    def add_input(self, value:float):
        self._pre_fire += value

    # Fire!
    def compute_activation(self):
        self._state += self._pre_fire
        self._state += self.bias
        self._state = self._activation(self._state)
        self._pre_fire = 0.

    @property
    def outgoing_edges(self):
        return self._outgoing_edges

    def __str__(self):
        return f"Node(label={self._label}, state={self._state})"


class Edge:
    def __init__(self, start_node, end_node, weight):
        self._start_node = start_node
        self._end_node = end_node
        self._weight = weight
        self._start_node.add_edge(self)
        self._id = f"{start_node.label}_{end_node.label}"  # Unique label for each edge
        
    @property
    def id(self):
        return self._id
    
    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def weight(self):
        return self._weight

    def __str__(self):
        return f"Edge(start={self._start_node.label}, end={self._end_node.label}, weight={self._weight})"


class Graph:
    def __init__(self, adjacency_dict, output_node_ids:list[int | str], max_steps:int=6, biases:dict[int, float]={}):
        self._nodes = {label: Node(label, activation=data['activation']) for label, data in adjacency_dict.items()}
        self._output_node_ids = output_node_ids
        self._edges = []
        self._max_steps = max_steps
        self._current_queue = deque()
        self._next_queue = deque()
        self._active_nodes = deque()
        self._adjacency_dict = adjacency_dict
        self._biases = biases

        # initialize nodes with states if provided
        for node_id, value in biases.items():
            if node_id in self._nodes:
                self._nodes[node_id].bias = value
    
        # Create edges with weights
        for start_label, node_data in adjacency_dict.items():
            for end_label, weight in node_data['edges'].items():
                self._edges.append(Edge(self._nodes[start_label], self._nodes[end_label], weight))


    def reset(self):
        for node in self._nodes.values():
            node.state = 0
        self._current_queue = deque()
        self._next_queue = deque()
        self._active_nodes = deque()


    def inference(self, input_values:dict, verbose:bool=False, memory:bool=True) -> dict:
        input_values = {self._nodes[label]: value for label, value in input_values.items()}
        # Create source edges for each input node
        source_edges = [Edge(Node("Source", state=value, activation=lambda x: x), node, weight=1) for node, value in input_values.items()]

        # Enqueue the source edges
        self._current_queue.extend(source_edges)
        self._next_queue = deque()
        # Initialize deque for nodes that received input
        self._active_nodes.extend([node for node, _ in input_values.items()])
        # First step is a dummy step to run the inputs into the input nodes
        step = -1
        # Input edges don't count
        energy_used = 0 - len(source_edges)

        while self._current_queue and step < self._max_steps:
            traversed_edges = set()  # Set to store traversed edges
            energy_used += len(self._current_queue)
            while self._current_queue:
                edge = self._current_queue.popleft()
                input_value = edge.start_node.state * edge.weight
                edge.end_node.add_input(input_value)

                # Add node to active_nodes
                if edge.end_node not in self._active_nodes:
                    self._active_nodes.append(edge.end_node)

                # Add only the outgoing edges that have not been traversed yet
                for out_edge in edge.end_node.outgoing_edges:
                    if out_edge.id not in traversed_edges:
                        self._next_queue.append(out_edge)
                        traversed_edges.add(out_edge.id) # Mark the edge as traversed

            # Apply activation to all active nodes
            while self._active_nodes:
                node = self._active_nodes.popleft()
                node.compute_activation()

            self._current_queue, self._next_queue = self._next_queue, self._current_queue
            step += 1
            if verbose:
                print(f'Step {step}, current state: {[node.state for node in self._nodes.values()]}')

        output_nodes = {nid: self._nodes[nid].state for nid in self._output_node_ids}

        energy_used /= len(self._nodes)
        if verbose:
            print(f"Total energy used: {energy_used}")
        
        if not memory:
            self.reset()
        
        return output_nodes


    def visualize(self, node_names:dict={}) -> None:
        dot = graphviz.Digraph()

        # Find input nodes (nodes with no incoming edges)
        all_nodes = set(self._adjacency_dict.keys())
        non_inputs = set()
        for _, value in self._adjacency_dict.items():
            for key, _ in value.get("edges").items():
                non_inputs.add(key)
                
        input_nodes = all_nodes - non_inputs
        
        for node, attr in self._adjacency_dict.items():
            node_name = node_names.get(node, "")
            activation_function = attr['activation'].__name__
            node_color = 'lightgreen' if node in self._output_node_ids else ('orange' if node in input_nodes else 'lightblue')
            dot.node(f"{node}", label=f"{node}\nActivation: {activation_function}\n{node_name}", style='filled', fillcolor=node_color)

        for node, attr in self._adjacency_dict.items():
            for neighbor, weight in attr['edges'].items():
                dot.edge(f"{node}", f"{neighbor}", label=f"Weight: {weight:.2f}", arrowhead="normal", arrowtail="normal")
        
        for output_node in self._output_node_ids:
            dot.node(f"{output_node}", pos="2,0!") #type: ignore

        dot.format = 'png'
        dot.render('graph_output', view=True)


    @staticmethod
    def genes_to_adjacency(genome, config):
        adjacency_matrix = {}

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values()]# if cg.enabled]
        all_nodes = set([k for k, _ in  genome.nodes.items()] + config.genome_config.input_keys)
       
        biases = {}
        for node_idx, node in genome.nodes.items():
            biases[node_idx] = node.bias
        
        for node in all_nodes:
            edges = {}
            for conn_key in connections:
                inode, onode = conn_key
                if node == inode:
                    cg = genome.connections[conn_key]
                    edges[onode] = cg.weight

            if node in config.genome_config.input_keys:
                activation_function = identity
            else:
                ng = genome.nodes[node]
                activation_function = config.genome_config.activation_defs.get(ng.activation)
            adjacency_matrix[node] = {
                'activation': activation_function,
                'edges': edges
            }
        return adjacency_matrix, config.genome_config.output_keys, biases

    @staticmethod
    def create(genome, config):
        matrix, output_nodes, biases = Graph.genes_to_adjacency(genome, config)
        return Graph(matrix, output_nodes, biases=biases)



def main():
    # Define adjacency matrix
    """
            H   A   B   C   D   E   F   G   I  Activation
    H       0   0   1   0   0   0   0   0   0     sigmoid
    A       0   0   1   1   0   0   0   0   0        relu
    B       0   0   0   0   1   1   0   0   0        tanh
    C       0   0   0   0   0   0   1   0   0     sigmoid
    D       0   0   0   1   0   0   0   0   1        tanh
    E       0   0   0   0   0   0   0   1   0    identity
    F       0   0   0   0   0   0   0   1   0     sigmoid
    G       0   0   0   0   0   0   0   0   0        tanh
    I       0   0   0   0   0   0   0   0   0    identity
    """
    
    adjacency_matrix = {
        'H': {'activation': sigmoid, 'edges': {'B': 1}},
        'A': {'activation': relu, 'edges': {'B': 1, 'C': 1}},
        'B': {'activation': tanh, 'edges': {'D': 1, 'E': 1}},
        'C': {'activation': sigmoid, 'edges': {'F': 1}},
        'D': {'activation': tanh, 'edges': {'C': 1, 'I': 1}},
        'E': {'activation': identity, 'edges': {'G': 1}},
        'F': {'activation': sigmoid, 'edges': {'G': 1}},
        'G': {'activation': tanh, 'edges': {}},
        'I': {'activation': identity, 'edges': {}},
    }

    # list of tuples mapping node labels to input values
    input_values = {'A': 1, 'H': 0.5}

    graph = Graph(adjacency_dict=adjacency_matrix, output_node_ids=['G', 'I'], max_steps=6)

    result = graph.inference(input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy"]
    print(f"Energy used: {energy:.2f}")
    for node in output_nodes:
        print(f'Output Node {node.label} final state: {node.state:.4f}')
    
    result = graph.inference(input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy"]
    print(f"Energy used: {energy:.2f}")
    for node in output_nodes:
        print(f'Output Node {node.label} final state: {node.state:.4f}')
    graph.reset()

    result = graph.inference(input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy"]
    print(f"Energy used: {energy:.2f}")
    for node in output_nodes:
        print(f'Output Node {node.label} final state: {node.state:.4f}')
    

if __name__ == "__main__":
    main()
