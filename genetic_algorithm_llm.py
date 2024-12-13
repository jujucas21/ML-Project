import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
import torch

# De-scale Weights for Inference
def reconstruct_weights(scaling_factors, remove_bit_values, quantization_dictionary):
    # Iterate through all layers in the model
    for i, (name, layer) in enumerate(quantization_dictionary.items()):
        # Initialize the weight matrix
        weight_matrix = layer
        
        if remove_bit_values[i] > 0 or scaling_factors[i] > 1:
            # Ensure that the layer is a integer tensor
            weight_matrix = layer.to(torch.int32)

            # Reconstruct the magnitude of the weights
            weight_matrix = weight_matrix << remove_bit_values[i]

            # De-scale the weights and biases
            weight_matrix = weight_matrix / float(scaling_factors[i])

        # Set the de-scaled weights back to the layer
        quantization_dictionary[name] = weight_matrix
           
    return quantization_dictionary

def quantize_weights(chromosome, quantization_dictionary):
    # Store values for future reconstruction of the activations
    scale_values = []
    remove_bit_values = []

    # Iterate through all layers in the model
    for i, (name, layer) in enumerate(quantization_dictionary.items()):
        weight_matrix = layer

        # No need to quantize float32 layers
        if chromosome[i] < 32:
            # Find the minimal value of the weights in the layer (ignoring zeros)
            min_value_inverse = 1 / torch.min(torch.abs(torch.where(weight_matrix != 0, weight_matrix, torch.ones(1, device=weight_matrix.device))))

            # Ensure that the layer is a integer tensor with low amount of information lost
            if min_value_inverse <= 1:
                min_value_inverse = 1
            elif min_value_inverse < 1000:
                min_value_inverse *= 1000

            # Scale the weights and convert to integer
            weight_matrix = (weight_matrix * min_value_inverse).to(torch.int32)

            # Find the maximum value of the weights in the layer
            max_value = torch.max(torch.abs(weight_matrix))

            # Find the minimum number of bits needed to represent the maximum value
            max_bits = int(torch.ceil(torch.log2(max_value.float()))) + 1

            # Compute the number of bits to remove
            remove_bits = max(max_bits - chromosome[i], 0)

            # Apply quantization: keep only the most significant 'chromosome[i]' bits
            weight_matrix = weight_matrix >> remove_bits

            # Store the values for future reconstruction
            remove_bit_values.append(remove_bits)
            scale_values.append(min_value_inverse.item())
        else:
            # No need to quantize float32 layers, store default values
            remove_bit_values.append(0)
            scale_values.append(1)

        # Update the dictionary with quantized weights
        quantization_dictionary[name] = weight_matrix

    return quantization_dictionary, scale_values, remove_bit_values

def clone_model(eval_dataset, model_name):
    distilled_bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def compute_accuracy(p):
        predictions, labels = p
        preds = predictions.argmax(axis=-1)  
        return {"accuracy": accuracy_score(labels, preds)}

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          
        eval_strategy="epoch",     
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        no_cuda=False        
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=distilled_bert_model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_accuracy
    )

    return distilled_bert_model, trainer

# Quantize the model and evaluate its performance on validation set
def quantize_model_and_evaluate_performance_llm(chromosome, val_dataset, model_name):
    # Clone the model to avoid modifying the original model
    cloned_model, trainer = clone_model(val_dataset, model_name)
    
    # Quantize the model based on the provided layer-wise bit values
    quantization_dictionary, scale_values, max_bit_values = quantize_weights(chromosome, cloned_model.state_dict())
    
    # Reconstruct weights to their original magnitude for inference
    quantized_layers = reconstruct_weights(scale_values, max_bit_values, quantization_dictionary)

    # Updated the weights of the model to the quantized ones
    cloned_model.load_state_dict(quantized_layers)

    # Evaluate the model on the validation set
    evaluation_information = trainer.evaluate()
    val_accuracy = evaluation_information['eval_accuracy']
    val_loss = evaluation_information['eval_loss']

    return val_loss, val_accuracy

# Evaluate the fitness of a chromosome
def evaluate_fitness(model, val_dataset, chromosome, v_q_ratio, model_name, min_accuracy):
    # Quantize the model and compute its validation loss
    val_loss, val_accuracy = quantize_model_and_evaluate_performance_llm(chromosome, val_dataset, model_name)

    # Compute the number of weights per layer
    weights_per_layer = [param.numel() for param in model.state_dict().values()]

    # Compute quantization score (we want a lower score!)
    quantization_score = sum([bit_width*n_weights for bit_width, n_weights in zip(chromosome, weights_per_layer)]/(np.sum(weights_per_layer)))/len(chromosome)
    
    # Compute the fitness score for this chromosome
    fitness = v_q_ratio*val_loss + quantization_score + 20*max(0, min_accuracy - val_accuracy)

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
            child[i] = int(np.random.choice(list(bit_widths), p=[0.001, 0.009, 0.09, 0.3, 0.3, 0.3]))
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
def perform_genetic_algorithm_llm(model, val_dataset, args):
    # Compute the amount of layers in the model    
    n_layers = len(model.state_dict())

    # Generate the initial population with probabilities certain probabilities for each gene
    population = [tuple(np.random.choice(args['bit_widths'], size=n_layers, p=args['initial_probabilities'])) for _ in range(args['population_size'])]
    
    # Add some radical cases to the initial population
    population[0] = [32] * n_layers
    population[1] = [16] * n_layers
    population[2] = [8] * n_layers
    population[3] = [4] * n_layers
    
    # Iterate through all generations
    for generation in range(args['generations']):
        # Print the number of the current generation
        print(f"Generation {generation + 1}/{args['generations']}")
        
        # Evaluate the fitness of the current population
        fitness_scores = [evaluate_fitness(model, val_dataset, chromosome, args['v_q_ratio'], args['model_name'], args['min_accuracy']) for chromosome in population]
        
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

