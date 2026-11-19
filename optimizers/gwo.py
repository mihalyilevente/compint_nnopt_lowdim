import torch


class GreyWolfOptimizer:
    """Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis, 2014), maximising fitness.

    alpha, beta and delta are the three best positions found so far (independent
    copies). The coefficient a decreases linearly from 2 to 0 over max_iter, moving
    the pack from exploration (|A| > 1) to exploitation (|A| < 1). The subspace has
    no natural box, so positions are not clipped to bounds.
    One fitness call per iteration plus the initial one, like PSO.
    """

    def __init__(self, population, dim, fitness_fn, device):
        if population < 3:
            raise ValueError("GWO needs a population of at least 3")
        self.population = population
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.device = device

    def train(self, max_iter=50, patience=10, init_scale=1.0):
        wolves = (torch.rand((self.population, self.dim), device=self.device) * 2 - 1) * init_scale
        fitness = self.fitness_fn(wolves).to(self.device)

        leader_values, idx = torch.topk(fitness, 3)
        leaders = wolves[idx].clone()           # rows: alpha, beta, delta
        leader_values = leader_values.clone()

        fitness_history = [leader_values[0].item()]
        patience_counter = 0

        for i in range(max_iter):
            a = 2 - 2 * i / max_iter
            new_wolves = torch.zeros_like(wolves)
            for leader in leaders:
                A = 2 * a * torch.rand_like(wolves) - a
                C = 2 * torch.rand_like(wolves)
                D = (C * leader - wolves).abs()
                new_wolves += leader - A * D
            wolves = new_wolves / 3

            fitness = self.fitness_fn(wolves).to(self.device)

            # New leaders: best three of the old leaders and the current pack.
            best_before = leader_values[0].item()
            pool_values = torch.cat([leader_values, fitness])
            pool = torch.cat([leaders, wolves])
            leader_values, idx = torch.topk(pool_values, 3)
            leaders = pool[idx].clone()
            leader_values = leader_values.clone()

            if leader_values[0].item() > best_before:
                patience_counter = 0
            else:
                patience_counter += 1

            fitness_history.append(leader_values[0].item())

            print(f"Iteration [{i+1}/{max_iter}]    Best fitness: {fitness_history[-1]:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return leaders[0].clone(), fitness_history
