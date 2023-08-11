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
        net = neat.nn.Graph.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            input_dict = dict(zip(config.genome_config.input_keys, xi))
            output = net.inference(input_dict)
            genome.fitness -= (output[0] - xo[0]) ** 2


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-spherical')

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
net = neat.nn.Graph.create(winner, config)
node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
net.visualize(node_names=node_names)

for xi, xo in zip(xor_inputs, xor_outputs):
    input_dict = dict(zip(config.genome_config.input_keys, xi))
    output = net.inference(input_dict)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
