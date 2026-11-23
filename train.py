from src.argument import parse_args
from src.utils import set_random_seeds
from models import LiProSTrainer
from pathlib import Path
import json
import os
import torch


METRIC_NAMES = [
    "train_acc", "val_acc", "test_acc",
    "train_precision", "val_precision", "test_precision",
    "train_recall", "val_recall", "test_recall",
    "train_f1", "val_f1", "test_f1",
]


def write_result(args, payload):
    if not args.result_json:
        return
    result_path = Path(args.result_json).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload["model"] = "LiProS"
    payload["seed"] = args.seed
    payload["config"] = vars(args)
    temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(result_path)
    print("[result] {}".format(result_path))


def main():
    args = parse_args()
    if args.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        print("[determinism] PyTorch strict deterministic algorithms ON")
    set_random_seeds(args.seed)
    torch.set_num_threads(2)

    trainer = LiProSTrainer(args)
    if args.mode == "bo":
        best_threshold, best_metrics = trainer.bayesian_optimization()
        write_result(args, {
            "mode": "bo",
            "best_threshold": float(best_threshold),
            "decision_threshold": float(getattr(trainer, "selected_decision_threshold", 0.5)),
            "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, best_metrics)},
        })
    else:
        result = trainer.train(args.confidence_threshold)
        write_result(args, {
            "mode": "single",
            "confidence_threshold": float(args.confidence_threshold),
            "decision_threshold": float(getattr(trainer, "selected_decision_threshold", 0.5)),
            "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, result[1:])},
        })


if __name__ == "__main__":
    main()
