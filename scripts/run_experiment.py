#!/usr/bin/env python3
"""Train DeepONet and FNO on exactly the same operator-learning dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from operator_learning_guide import FNO1D, DeepONet1D, count_parameters
from operator_learning_guide.data import generate_poisson_dataset
from operator_learning_guide.training import fit_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = choose_device(args.device)
    dataset = generate_poisson_dataset(
        n_train=config["n_train"],
        n_validation=config["n_val"],
        n_test=config["n_test"],
        n_grid=config["n_grid"],
        smooth_modes=config["smooth_modes"],
        log_coefficient_scale=config["log_coefficient_scale"],
        seed=config["seed"],
    )

    model_specs = {
        "deeponet": DeepONet1D(n_sensors=config["n_grid"], **config["deeponet"]),
        "fno": FNO1D(**config["fno"]),
    }
    summary: dict[str, object] = {
        "config": config,
        "device": str(device),
        "models": {},
    }

    for name, model in model_specs.items():
        seed_everything(config["seed"])
        started = time.perf_counter()
        result = fit_model(
            model,
            train_coefficients=dataset.train.coefficients,
            train_solutions=dataset.train.solutions,
            validation_coefficients=dataset.validation.coefficients,
            validation_solutions=dataset.validation.solutions,
            test_coefficients=dataset.test.coefficients,
            test_solutions=dataset.test.solutions,
            grid=dataset.grid,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            device=device,
        )
        summary["models"][name] = {
            "parameters": count_parameters(model),
            "seconds": time.perf_counter() - started,
            "history": result.history,
            "test_metrics": result.test_metrics,
        }
        print(f"{name}: {result.test_metrics}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
