import hydra
import os
import json
import torch
import mlflow
import wandb

from omegaconf import DictConfig
from hydra.utils import instantiate

from surgical_copilot.bench.BenchmarkEngine import BenchmarkEngine


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="config"
)
def run_benchmark(cfg: DictConfig):

    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model).to(device)
    dataset = instantiate(cfg.data)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.robustness.batch_size,
        shuffle=False,
        num_workers=cfg.robustness.num_workers
    )

    model.eval()

    run_name = cfg.model._target_.split(".")[-1]

    mlflow.set_experiment(cfg.logging.project)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "model": cfg.model._target_,
        "seed": cfg.seed,
        "device": str(device)
    })

    wandb.init(
        project=cfg.logging.project,
        name=run_name,
        config=dict(cfg)
    )

    engine = BenchmarkEngine(model, loader, cfg, device)
    results = engine.run()

    os.makedirs("metrics", exist_ok=True)
    out_path = f"report/benchmark/{run_name}_benchmark.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    summary = {
        "dice_mean": sum(r["dice_mean"] for r in results) / len(results),
        "fps_mean": sum(r["fps_mean"] for r in results) / len(results),
    }

    mlflow.log_metrics(summary)
    mlflow.log_artifact(out_path)

    wandb.log(summary)
    wandb.save(out_path)

    mlflow.end_run()
    wandb.finish()

    print(f"[DONE] saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()