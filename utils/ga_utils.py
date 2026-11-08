import math

import torch

def truncation_selection(fitness, keep_frac=0.5):
    """Keep the top `keep_frac` of the population (highest fitness) as parents.

    Survivors are repeated to refill the population and shuffled, so adjacent
    parents (paired by arithmetic_crossover) are random survivor pairs.
    """
    if not 0 < keep_frac <= 1:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    population = fitness.shape[0]
    keep = max(1, math.ceil(population * keep_frac))

    survivors = torch.argsort(fitness, descending=True)[:keep]
    parents = survivors.repeat(math.ceil(population / keep))[:population]
    parents = parents[torch.randperm(population, device=fitness.device)]

    return parents



def tournament_selection(fitness, device, group_size=2):
    population = fitness.shape[0]
    indices = torch.randint(0, population, (population, group_size), device=device)
    winners = torch.argmax(fitness[indices], 1)
    parents = indices[torch.arange(0, population, device=device), winners]

    return parents



def arithmetic_crossover(genomes, parents, device):
    if genomes.shape[0] % 2 != 0:
        raise ValueError(f"arithmetic_crossover needs an even population, got {genomes.shape[0]}")
    alpha = torch.rand((genomes.shape[0], 1), device=device)
    new_genomes = torch.cat((
        alpha[0::2] * genomes[parents[0::2]] + (1 - alpha)[0::2] * genomes[parents[1::2]],
        (1 - alpha)[1::2] * genomes[parents[0::2]] + alpha[1::2] * genomes[parents[1::2]]
    ), 0)

    return new_genomes



def gaussian_mutation(genomes, device, sigma=0.1):
    return genomes + torch.randn_like(genomes, device=device) * sigma
