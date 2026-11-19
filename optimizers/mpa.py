import math

import torch


def levy_steps(shape, beta, device):
    """Levy-stable steps via Mantegna's algorithm (as in the reference MPA code)."""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = torch.randn(shape, device=device) * sigma
    v = torch.randn(shape, device=device)
    return u / v.abs().pow(1 / beta)


class MarinePredatorsOptimizer:
    """Marine Predators Algorithm (Faramarzi et al., 2020), maximising fitness.

    Follows the authors' reference implementation: three phases by iteration
    (Brownian prey steps; half Levy / half Brownian; Levy predator steps), a top
    predator (Elite) and marine memory saving after each evaluation, then FADs /
    eddy formation. Two fitness calls per iteration. The subspace has no natural
    box: FADs jumps use [-init_scale, init_scale] as their range and positions
    are not clipped.
    """

    def __init__(self, population, dim, fitness_fn, device):
        if population < 2:
            raise ValueError("MPA needs a population of at least 2")
        self.population = population
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.device = device

    def train(self, max_iter=50, patience=10, init_scale=1.0, fads=0.2, p=0.5):
        n, d, device = self.population, self.dim, self.device
        half = n // 2
        prey = (torch.rand((n, d), device=device) * 2 - 1) * init_scale

        state = dict(top=torch.zeros(d, device=device), top_value=-float("inf"), prey_old=None, fit_old=None)

        def evaluate_and_remember(prey):
            fitness = self.fitness_fn(prey).to(device)
            idx = torch.argmax(fitness).item()
            if fitness[idx].item() > state["top_value"]:
                state["top"] = prey[idx].clone()
                state["top_value"] = fitness[idx].item()
            # Marine memory saving: keep a prey's previous position if it was better.
            if state["prey_old"] is not None:
                keep = state["fit_old"] > fitness
                prey = torch.where(keep[:, None], state["prey_old"], prey)
                fitness = torch.where(keep, state["fit_old"], fitness)
            state["prey_old"], state["fit_old"] = prey.clone(), fitness.clone()
            return prey

        fitness_history = []
        patience_counter = 0

        for i in range(max_iter):
            best_before = state["top_value"]
            prey = evaluate_and_remember(prey)

            elite = state["top"].expand(n, d)
            cf = (1 - i / max_iter) ** (2 * i / max_iter)
            rl = 0.05 * levy_steps((n, d), 1.5, device)
            rb = torch.randn((n, d), device=device)
            r = torch.rand((n, d), device=device)

            if i < max_iter / 3:                      # phase 1: prey moves faster (Brownian)
                prey = prey + p * r * (rb * (elite - rb * prey))
            elif i < 2 * max_iter / 3:                # phase 2: half Levy (prey), half Brownian (predator)
                new_prey = prey.clone()
                new_prey[:half] = prey[:half] + p * r[:half] * (rl[:half] * (elite[:half] - rl[:half] * prey[:half]))
                new_prey[half:] = elite[half:] + p * cf * (rb[half:] * (rb[half:] * elite[half:] - prey[half:]))
                prey = new_prey
            else:                                     # phase 3: predator moves with Levy steps
                prey = elite + p * cf * (rl * (rl * elite - prey))

            prey = evaluate_and_remember(prey)

            # Eddy formation and FADs effect.
            if torch.rand(()).item() < fads:
                mask = (torch.rand((n, d), device=device) < fads).float()
                jumps = -init_scale + torch.rand((n, d), device=device) * 2 * init_scale
                prey = prey + cf * jumps * mask
            else:
                rr = torch.rand(()).item()
                prey = prey + (fads * (1 - rr) + rr) * (prey[torch.randperm(n, device=device)]
                                                        - prey[torch.randperm(n, device=device)])

            if state["top_value"] > best_before:
                patience_counter = 0
            else:
                patience_counter += 1

            fitness_history.append(state["top_value"])

            print(f"Iteration [{i+1}/{max_iter}]    Best fitness: {state['top_value']:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return state["top"].clone(), fitness_history
