import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# De-scale Weights for Inference
def reconstruct_weights(model, scaling_factors, remove_bit_values, quantization_layers):
    # Iterate through all layers in the model
    for i, layer in enumerate(quantization_layers):
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            # Check if the layer has weights and biases
            if len(layer.get_weights()) == 2:
                # Extract the weights and biases from the layer
                weight_matrix, bias = layer.get_weights()

                if remove_bit_values[i] > 0:
                    # Reconstruct the magnitude of the weights and biases
                    weight_matrix = np.int32(weight_matrix) << remove_bit_values[i]
                    bias = np.int32(bias) << remove_bit_values[i]

                    # De-scale the weights and biases
                    weight_matrix = weight_matrix / float(scaling_factors[i])
                    bias = bias / float(scaling_factors[i])

                # Set the de-scaled weights back to the layer
                layer.set_weights([weight_matrix, bias])
            # Check if the layer has only weights
            elif len(layer.get_weights()) == 1:
                # Extract the weights from the layer
                weight_matrix = layer.get_weights()[0]

                if remove_bit_values[i] > 0:
                    # Reconstruct the magnitude of the weights
                    weight_matrix = np.int32(weight_matrix) << remove_bit_values[i]

                    # De-scale the weights
                    weight_matrix = weight_matrix / float(scaling_factors[i])

                # Set the de-scaled weights back to the layer
                layer.set_weights([weight_matrix])
        
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # Extract the gamma, beta, moving mean, and moving variance from the layer
            gamma, beta, moving_mean, moving_variance = layer.get_weights()

            if remove_bit_values[i] > 0:
                # Reconstruct the magnitude of the gamma, beta, moving mean, and moving variance
                gamma = np.int32(gamma) << remove_bit_values[i]
                beta = np.int32(beta) << remove_bit_values[i]
                moving_mean = np.int32(moving_mean) << remove_bit_values[i]
                moving_variance = np.int32(moving_variance) << remove_bit_values[i]

                # De-scale the gamma, beta, moving mean, and moving variance
                gamma = gamma / float(scaling_factors[i])
                beta = beta / float(scaling_factors[i])
                moving_mean = moving_mean / float(scaling_factors[i])
                moving_variance = moving_variance / float(scaling_factors[i])

            # Set the de-scaled gamma, beta, moving mean, and moving variance back to the layer
            layer.set_weights([gamma, beta, moving_mean, moving_variance])    
    return model

# Quantize the weights of the model
def quantize_weights(model, chromosome, quantization_layers):
    # Store values for future reconstruction of the activations
    scale_values = []
    remove_bit_values = []

    # Iterate through all layers in the model
    for i, layer in enumerate(quantization_layers):
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            # Get the weights of the current layer
            if len(layer.get_weights()) == 2:  
                # Layer has weights and biases
                weight_matrix, bias = layer.get_weights()
            elif len(layer.get_weights()) == 1:  
                # Layer has only weights
                weight_matrix = layer.get_weights()[0]
                bias = None

            # No need to quantize float32 layers
            if chromosome[i] < 32:
                # Find the minimal value of the weights in the layer
                min_value_inverse = 1/np.min([np.min(np.abs(np.where(weight_matrix != 0, weight_matrix, 1))), np.min(np.abs(np.where(bias != 0, bias, 1)))])  

                # Scale the weights in order to convert them to integers
                weight_matrix = np.int32(weight_matrix * float(min_value_inverse))
                if bias is not None: 
                    bias = np.int32(bias * float(min_value_inverse))

                # Find the maximum value of the weights in the layer
                if bias is not None: 
                    max_value = np.max([np.max(np.abs(weight_matrix)), np.max(np.abs(bias))])
                else: 
                    max_value = np.max(np.abs(weight_matrix))

                # Find the minimum amount of bits needed to represent the maximum value
                max_bits = int(np.ceil(np.log2(max_value))) + 1

                # Compute the amout of bits that need to be removed
                remove_bits = max(max_bits - chromosome[i], 0)
                
                # Apply quantization: keep only the most significant 'chromosome[i]' bits
                weight_matrix = weight_matrix >> remove_bits 
                bias = bias >> remove_bits 

                # Store the values for future reconstruction
                remove_bit_values.append(remove_bits)
                scale_values.append(min_value_inverse)
            else:
                # No need to quantize float32 layers, store default values
                remove_bit_values.append(0)
                scale_values.append(1)

            # Set the quantized weights back to the layer
            if bias is not None:
                layer.set_weights([weight_matrix, bias])
            else:
                layer.set_weights([weight_matrix])
        
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # Get the four weights: gamma, beta, moving mean, and moving variance
            gamma, beta, moving_mean, moving_variance = layer.get_weights()

            # Quantize each parameter individually if necessary
            if chromosome[i] < 32:
                min_value_inverse = 1/np.min([np.min(np.abs(np.where(gamma != 0, gamma, 1))),np.min(np.abs(np.where(beta != 0, beta, 1))),np.min(np.abs(np.where(moving_mean != 0, moving_mean, 1))), np.min(np.abs(np.where(moving_variance != 0, moving_variance, 1)))])  
                gamma = np.int32(gamma * min_value_inverse)
                beta = np.int32(beta * min_value_inverse)
                moving_mean = np.int32(moving_mean * min_value_inverse)
                moving_variance = np.int32(moving_variance * min_value_inverse)

                max_value = np.max([np.max(np.abs(gamma)), np.max(np.abs(beta)), 
                                    np.max(np.abs(moving_mean)), np.max(np.abs(moving_variance))])
                max_bits = int(np.ceil(np.log2(max_value))) + 1
                remove_bits = max(max_bits - chromosome[i], 0)

                # Apply quantization
                gamma = gamma >> remove_bits
                beta = beta >> remove_bits
                moving_mean = moving_mean >> remove_bits
                moving_variance = moving_variance >> remove_bits

                # Store scale and removed bits
                scale_values.append(min_value_inverse)
                remove_bit_values.append(remove_bits)
            else:
                scale_values.append(1)
                remove_bit_values.append(0)

            # Set quantized weights back to the layer
            layer.set_weights([gamma, beta, moving_mean, moving_variance])
    
    return model, scale_values, remove_bit_values

# Quantize the model and evaluate its performance on validation set
def quantize_model_and_evaluate_performance_mnist(model, chromosome, criterion, val_dataset):
    # Clone the model to avoid modifying the original model
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())

    # Extract layers that have weights
    quantization_layers = [layer for layer in cloned_model.layers if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization))]
        
    # Quantize the model based on the provided layer-wise bit values
    model_quantized, scale_values, max_bit_values = quantize_weights(cloned_model, chromosome, quantization_layers)
    
    # Reconstruct weights to their original magnitude for inference
    dequantized_inference_model = reconstruct_weights(model_quantized, scale_values, max_bit_values, quantization_layers)

    # Evaluate the performance of the dequantized model on the validation set
    predictions = dequantized_inference_model.predict(val_dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in val_dataset])
    val_accuracy = accuracy_score(true_labels, predicted_labels)   
    val_loss = criterion(true_labels, predictions).numpy() 

    return val_loss, val_accuracy, dequantized_inference_model

# Evaluate the fitness of a chromosome
def evaluate_fitness(model, val_dataset, criterion, chromosome, v_q_ratio, min_accuracy):
    # Quantize the model and compute its validation loss
    val_loss, val_accuracy, _ = quantize_model_and_evaluate_performance_mnist(model, chromosome, criterion, val_dataset)

    # Compute the amount of weights per layer. 
    quantization_layers = [layer for layer in model.layers.copy() if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization))]
    weights_per_layer = [
        sum(np.prod(w.shape) for w in layer.get_weights())
        if layer.get_weights() else 0
        for layer in quantization_layers
    ]   

    # Compute quantization score (we want a lower score!)
    quantization_score = sum([bit_width*n_weights for bit_width, n_weights in zip(chromosome, weights_per_layer)]/(np.sum(weights_per_layer)*len(chromosome))) 
    
    # Compute the fitness score for this chromosome
    fitness = v_q_ratio*val_loss + quantization_score + 20*max(0, min_accuracy - val_accuracy)

    # Make sure the chromosome is a tuple for consistency
    chromosome = tuple(chromosome)

    # Print the accuracy, quantization score and final fitness of the current chromosome
    print('Val-loss:', val_loss, 'accuracy:', val_accuracy, 'quantization score:', quantization_score, 'fitness:', fitness)
    print('Amount of 1s:', chromosome.count(1), 'Amount of 2s:', chromosome.count(2), 'Amount of 4s:', chromosome.count(4), 'Amount of 8s:', chromosome.count(8), 'Amount of 16s:', chromosome.count(16), 'Amount of 32s:', chromosome.count(32))

    return fitness, val_loss, val_accuracy

# GA Operations
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)) 
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return tuple(child)

def mutate(child, mutation_rate, bit_widths):
    child = list(child)
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = int(np.random.choice(list(bit_widths), p=[0.05, 0.19, 0.19, 0.19, 0.19, 0.19]))
    return tuple(child)

# Construct the next generation
def construct_new_population(sorted_population, args):
    # Make sure all elements of chromosomes are integers
    sorted_population = [tuple(map(int, chromosome)) for chromosome in sorted_population]

    # Initialize the generation by keeping the top chromosomes: elitism
    next_generation_population = sorted_population[:args['elitism_amount']]

    # Add some elements that are just mutated versions of the top chromosomes
    for chromosome in sorted_population[:args['elitism_amount']]:
        mutated_chromosome = mutate(chromosome, 0.3, args['bit_widths'])
        next_generation_population.append(mutated_chromosome)

    # Compute the next generation by applying crossover and mutation operations
    while len(next_generation_population) < args['population_size']:
        # Select two parents from the top chromosomes
        parent1, parent2 = np.random.choice(range(0, args['crossover_top']), size=2, replace=False)
        parent1, parent2 = sorted_population[:args['crossover_top']][parent1], sorted_population[:args['crossover_top']][parent2]

        # Apply crossover and mutation to create a new child
        child = crossover(parent1, parent2)
        child = mutate(child, args['mutation_rate'], args['bit_widths'])

        # Add the new chromosme to the next generation
        next_generation_population.append(child)
    
    return next_generation_population

# Genetic Algorithm
def perform_genetic_algorithm_mnist(model, val_dataset, args):
    # Compute the amount of layers in the model
    n_layers = len([layer for layer in model.layers if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization))])

    # Initialize the population with random chromosomes
    population = [np.random.choice(args['bit_widths'], size=n_layers) for _ in range(args['population_size'])]
    
    # Add certain cases to the initial population
    population[0] = [2] * n_layers
    population[1] = [4] * n_layers  
    population[2] = [8] * n_layers  
    population[3] = [16] * n_layers  
    population[4] = [32] * n_layers  
    
    # Iterate through all generations
    for generation in range(args['generations']):
        # Print the number of the current generation
        print(f"Generation {generation + 1}/{args['generations']}")

        # Evaluate the fitness of the current population
        fitness_scores = [evaluate_fitness(model, val_dataset, args['loss_function'], chromosome, args['v_q_ratio'], args['min_accuracy']) for chromosome in population]
        
        # Sort the population based on fitness scores (one element is a tuple of fitness score, validation loss and chromosme)
        sorted_population = [(x, y, z, q) for x, y, z, q in sorted(zip([x for x, _, _ in fitness_scores], [x for _, x, _ in fitness_scores], [x for _, _, x in fitness_scores], [tuple(chromosome) for chromosome in population]))]
                
        # Update the fittest chromosome, and prepare to print
        best_fitness = sorted_population[0][0]
        best_loss = sorted_population[0][1]
        best_accuracy = sorted_population[0][2]
        best_chromosome = sorted_population[0][3]

        # Print the best fitness, validation and chromosome of the current generation
        print(f"Best fitness: {best_fitness}")
        print(f"Best accuracy: {best_accuracy}")
        print(f"Corresponding validation loss: {best_loss}")   
        print(f"Corresponding chromosome: {best_chromosome}\n")
        
        # Update the current population
        population = construct_new_population([chromosome for _, _, _, chromosome in sorted_population], args)
    
    return best_chromosome, best_fitness, best_accuracy, best_loss