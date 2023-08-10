"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import neat

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = 4.0
        matrix, output_ids, biases = neat.nn.spherical.Graph.genes_to_adjacency(genome, config)
        net = neat.nn.spherical.Graph(matrix, output_ids, biases=biases)
        for xi, xo in zip(xor_inputs, xor_outputs):
            input_dict = dict(zip(config.genome_config.input_keys, xi))
            try: # sometimes the pruning I guess removes an input
                output = net.inference(input_dict)
            except:
                output = [abs(1-xo[0])]
            genome.fitness -= abs(output[0] - xo[0])


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
matrix, output_ids, biases = neat.nn.spherical.Graph.genes_to_adjacency(winner, config)
winner_net = neat.nn.spherical.Graph(matrix, output_ids, biases=biases)
winner_net.visualize()

for xi, xo in zip(xor_inputs, xor_outputs):
    input_dict = dict(zip(config.genome_config.input_keys, xi))
    output = winner_net.inference(input_dict)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
