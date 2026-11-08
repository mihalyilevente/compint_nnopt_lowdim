"""Bounded environment check: imports, 32 MNIST samples and one Adam step/model."""
import ast
import importlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
import pandas as pd
import torch
from IPython.core.interactiveshell import InteractiveShell
from torch.utils.data import DataLoader
from models.mnist_mlp import MNIST_MLP
from models.lenet5 import LeNet5
from models.vector_mlp import VectorMLP
from utils.mnist_dataloader import MNISTDataset
from utils.fitness_function import FitnessFunction
from utils.train_model import _train_one_epoch, _validate


def main():
    torch.set_num_threads(2)
    modules = []
    for folder in ("models", "optimizers", "utils"):
        for path in sorted((ROOT / folder).glob("*.py")):
            name = f"{folder}.{path.stem}"
            importlib.import_module(name)
            modules.append(name)
    # Compile every transformed notebook cell; execute ONLY top-level import statements.
    shell = InteractiveShell.instance()
    notebook_checks = {}
    for path in ROOT.glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        scope = {}
        count = 0
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            tree = ast.parse(shell.input_transformer_manager.transform_cell(cell.source))
            compile(tree, f"{path.name}:cell{index}", "exec")
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
                    count += 1
        notebook_checks[path.name] = count
    frame = pd.read_csv(ROOT / "data/mnist/mnist_train.csv", nrows=32)
    with tempfile.TemporaryDirectory() as temporary:
        subset = Path(temporary) / "subset.csv"
        frame.to_csv(subset, index=False)
        dataset = MNISTDataset(subset)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    results = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for name, factory in [("MLP", lambda: MNIST_MLP([200, 200])),
                              ("LeNet5", LeNet5),
                              ("VectorMLP", lambda: VectorMLP(MNIST_MLP([200, 200]), 8)),
                              ("VectorLeNet5", lambda: VectorMLP(LeNet5(), 8))]:
            torch.manual_seed(42)
            model = factory().to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in params]
            buffers_before = {n: b.clone() for n, b in model.named_buffers()}
            optimizer = torch.optim.Adam(params, lr=0.001)
            loss, acc = _train_one_epoch(model, device, loader, torch.nn.CrossEntropyLoss(), optimizer)
            assert torch.isfinite(torch.tensor(loss))
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert any(not torch.equal(a, b) for a, b in zip(before, params))
            for n, b in model.named_buffers():
                torch.testing.assert_close(b, buffers_before[n], rtol=0, atol=0)
            val_loss, _ = _validate(model, device, loader, torch.nn.CrossEntropyLoss())
            assert torch.isfinite(torch.tensor(val_loss))
            if isinstance(model, VectorMLP):
                assert all(p.grad is None for p in model.model.parameters())
                candidates = torch.stack([model.lower_weights.detach(), torch.zeros_like(model.lower_weights)])
                fitness = FitnessFunction(model, loader, 1, device)(candidates)
                x, y = next(iter(loader))
                with torch.no_grad():
                    serial = torch.stack([(model.parallel_eval(z, x.to(device)).argmax(1) == y.to(device)).float().mean() for z in candidates])
                torch.testing.assert_close(fitness, serial)
            results.append(dict(device=device, model=name, samples=32, steps=1, loss=loss, accuracy=acc))
    print(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         imported_modules=modules, notebook_import_statements=notebook_checks,
                         results=results), indent=2))


if __name__ == "__main__":
    main()
