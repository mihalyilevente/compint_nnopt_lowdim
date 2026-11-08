import torch
import torch.nn.functional as F
from torch.func import vmap


class FitnessFunction:
    """Population fitness on a fixed data loader (higher is better).

    metric="accuracy": fraction of correctly classified samples (original behaviour).
    metric="neg_ce": negative mean cross-entropy, a smoother objective.
    """

    def __init__(self, model, dataloader, chunk_size, device, metric="accuracy"):
        if metric not in ("accuracy", "neg_ce"):
            raise ValueError(f"unknown metric {metric!r}")
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.chunk_size = chunk_size
        self.metric = metric

        self.model.eval()


    def __call__(self, parameters):
        parameters = parameters.to(self.device)
        score = torch.zeros((parameters.shape[0],), device=self.device)

        total = 0
        with torch.no_grad():
            parallel_eval_fn = vmap(self.model.parallel_eval, in_dims=(0, None), chunk_size=self.chunk_size)
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)

                outputs = parallel_eval_fn(parameters, data)  # (population, batch, classes)
                if self.metric == "accuracy":
                    score += torch.sum(outputs.argmax(2) == target, 1)
                else:
                    targets = target.expand(outputs.shape[0], -1)
                    score -= F.cross_entropy(outputs.permute(0, 2, 1), targets, reduction="none").sum(1)
                total += target.shape[0]

        return score / total
