import torch

class ParticleSwarmOptimizer:
    def __init__(self, population, dim, fitness_fn, device):
        self.population = population
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.device = device

    def train(self, w=0.7, cp=1.5, cg=1.5, max_iter=50, patience=10):
        positions = torch.rand((self.population, self.dim), device=self.device) * 2 - 1
        velocities = torch.rand((self.population, self.dim), device=self.device) * 4 - 2

        best_positions = positions.clone()
        best_values = self.fitness_fn(positions).to(self.device)

        # Global best is an independent snapshot (clone / Python float), never a view
        # into best_positions / best_values, which are updated in place below.
        best_idx = torch.argmax(best_values).item()
        global_best_position = best_positions[best_idx].clone()
        global_best_value = best_values[best_idx].item()

        fitness_history = [global_best_value]

        patience_counter = 0

        for i in range(max_iter):
            # Update particles
            rp = torch.rand((self.population, self.dim), device=self.device)
            rg = torch.rand((self.population, self.dim), device=self.device)

            velocities = w * velocities + cp * rp * (best_positions - positions) + cg * rg * (global_best_position - positions)
            positions = positions + velocities

            # Calculate fitness
            fitness = self.fitness_fn(positions).to(self.device)

            # Update personal best records
            mask = fitness > best_values
            best_positions[mask, :] = positions[mask, :]
            best_values[mask] = fitness[mask]

            # Update global best; every iteration without a global improvement counts
            # toward patience, including iterations where no particle improved.
            best_idx = torch.argmax(best_values).item()
            if best_values[best_idx].item() > global_best_value:
                global_best_position = best_positions[best_idx].clone()
                global_best_value = best_values[best_idx].item()
                patience_counter = 0
            else:
                patience_counter += 1

            fitness_history.append(global_best_value)

            print(f"Iteration [{i+1}/{max_iter}]    Best fitness: {global_best_value:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return global_best_position, fitness_history
