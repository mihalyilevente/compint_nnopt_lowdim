"""Tests for the Li et al. (2018) subspace wrapper. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lenet5 import LeNet5
from models.mnist_mlp import MNIST_MLP
from models.vector_mlp import VectorMLP


def small_net():
    return nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))


class LiSubspace(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(8, 784)

    def test_z_zero_reproduces_wrapped_model_exactly(self):
        for factory in (lambda: MNIST_MLP([200, 200]), LeNet5):
            base = factory()
            expected = base(self.x)
            wrapper = VectorMLP(base, lower_dim=16, proj_seed=1)
            torch.testing.assert_close(wrapper(self.x), expected)

    def test_theta0_is_model_init(self):
        base = MNIST_MLP([200, 200])
        flat = torch.cat([p.detach().reshape(-1) for p in base.parameters()])
        wrapper = VectorMLP(base, lower_dim=8, proj_seed=1)
        self.assertTrue(torch.equal(wrapper.bias, flat))

    def test_columns_have_unit_norm(self):
        wrapper = VectorMLP(MNIST_MLP([200, 200]), lower_dim=32, proj_seed=1)
        torch.testing.assert_close(wrapper.proj_matrix.norm(dim=0), torch.ones(32))

    def test_projection_seed_is_independent_of_model_init(self):
        torch.manual_seed(1)
        a = VectorMLP(small_net(), lower_dim=5, proj_seed=7)
        torch.manual_seed(2)
        b = VectorMLP(small_net(), lower_dim=5, proj_seed=7)
        c = VectorMLP(small_net(), lower_dim=5, proj_seed=8)
        self.assertTrue(torch.equal(a.proj_matrix, b.proj_matrix))
        self.assertFalse(torch.equal(a.bias, b.bias))
        self.assertFalse(torch.equal(a.proj_matrix, c.proj_matrix))

    def test_global_seed_reproduces_when_no_proj_seed(self):
        torch.manual_seed(3)
        a = VectorMLP(small_net(), lower_dim=5)
        torch.manual_seed(3)
        b = VectorMLP(small_net(), lower_dim=5)
        self.assertTrue(torch.equal(a.proj_matrix, b.proj_matrix))
        self.assertTrue(torch.equal(a.bias, b.bias))

    def test_only_z_is_trainable(self):
        wrapper = VectorMLP(small_net(), lower_dim=5, proj_seed=1)
        trainable = [n for n, p in wrapper.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, ["lower_weights"])
        self.assertTrue(torch.equal(wrapper.lower_weights, torch.zeros(5)))

    def test_gradient_matches_finite_differences(self):
        wrapper = VectorMLP(small_net().double(), lower_dim=5, proj_seed=1)
        x = torch.rand(6, 4, dtype=torch.float64)
        z = torch.randn(5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda w: wrapper.parallel_eval(w, x).sum(), (z,)))

    def test_vmap_matches_serial(self):
        wrapper = VectorMLP(MNIST_MLP([200, 200]), lower_dim=16, proj_seed=1)
        zs = torch.randn(4, 16)
        with torch.no_grad():
            batched = vmap(wrapper.parallel_eval, in_dims=(0, None))(zs, self.x)
            serial = torch.stack([wrapper.parallel_eval(z, self.x) for z in zs])
        torch.testing.assert_close(batched, serial)

    def test_eval_and_train_propagate(self):
        wrapper = VectorMLP(small_net(), lower_dim=5, proj_seed=1)
        wrapper.eval()
        self.assertFalse(wrapper.model.training)
        wrapper.train()
        self.assertTrue(wrapper.model.training)

    def test_checkpoint_round_trip(self):
        a = VectorMLP(small_net(), lower_dim=5, proj_seed=1)
        with torch.no_grad():
            a.lower_weights.copy_(torch.randn(5))
        b = VectorMLP(small_net(), lower_dim=5, proj_seed=2)
        b.load_state_dict(a.state_dict())
        x = torch.rand(3, 4)
        self.assertTrue(torch.equal(a(x), b(x)))

    def test_legacy_mode_keeps_old_behaviour(self):
        base = MNIST_MLP([200, 200])
        wrapper = VectorMLP(base, lower_dim=8, proj_seed=1, legacy=True)
        norms = wrapper.proj_matrix.norm(dim=0)
        self.assertTrue((norms > 100).all())  # unnormalised: ~sqrt(D) = ~446
        self.assertFalse(torch.equal(wrapper(self.x), base(self.x)))


if __name__ == "__main__":
    unittest.main()
