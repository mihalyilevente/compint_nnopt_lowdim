import torch.nn as nn
from torch.func import functional_call
import torch


class VectorMLP(nn.Module):
    """Subspace wrapper: theta = theta0 + A z  (Li et al. 2018, dense projection).

    - theta0 (buffer ``bias``): the wrapped model's own initialised weights, frozen.
    - A (buffer ``proj_matrix``): fixed Gaussian D x d matrix with unit-length columns.
    - z (``lower_weights``): the only trainable parameter, initialised to zero,
      so the wrapper starts exactly at the wrapped model.

    ``proj_seed`` makes A reproducible independently of the model's init seed, so
    different optimisers can share the same subspace. ``legacy=True`` restores the
    original behaviour (random theta0 ~ N(0, 1/D), unnormalised A) for comparison
    with old notebook results only.
    """

    def __init__(self, model, lower_dim=500, proj_seed=None, legacy=False):
        super().__init__()

        self.model = model
        self.lower_dim = lower_dim
        self.legacy = legacy

        for param in model.parameters():
            param.requires_grad_(False)
        theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
        self.num_weights = theta0.numel()

        generator = torch.Generator()
        if proj_seed is None:
            # Fall back to the global RNG so torch.manual_seed() still makes runs reproducible.
            proj_seed = int(torch.randint(0, 2**62, (1,)).item())
        generator.manual_seed(proj_seed)
        self.proj_seed = proj_seed

        proj = torch.randn((self.num_weights, lower_dim), generator=generator, dtype=theta0.dtype)
        if legacy:
            theta0 = torch.randn((self.num_weights,), generator=generator, dtype=theta0.dtype) / (self.num_weights ** 0.5)
        else:
            proj = proj / proj.norm(dim=0, keepdim=True)

        # Non-trainable (buffer names kept for compatibility with existing checkpoints).
        self.register_buffer("proj_matrix", proj.to(theta0.device))
        self.register_buffer("bias", theta0)

        # Trainable parameters
        self.lower_weights = nn.Parameter(torch.zeros((lower_dim,), dtype=theta0.dtype, device=theta0.device))

    def _full_params(self, weights):
        higher_weights = self.bias + self.proj_matrix @ weights

        params = {}
        i = 0
        for name, param in self.model.named_parameters():
            params[name] = higher_weights[i:i+param.numel()].view(param.shape)
            i += param.numel()
        return params

    def forward(self, x):
        return functional_call(self.model, self._full_params(self.lower_weights), (x,))

    def parallel_eval(self, weights, x):
        return functional_call(self.model, self._full_params(weights), (x,))
