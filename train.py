import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from src.argument import parse_args


logger = logging.getLogger(__name__)


def main(*, command_line=False):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.deterministic_algorithms:
        if command_line and "PYTHONHASHSEED" not in os.environ:
            env = {**os.environ, "PYTHONHASHSEED": str(args.seed % (2**32))}
            return subprocess.call([sys.executable, *sys.orig_argv[1:]], env=env)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from models import LiProSTrainer
    from src.utils import METRIC_NAMES, atomic_output, set_random_seeds

    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("[determinism] PyTorch warn-only mode | PYTHONHASHSEED=%s",
                    os.environ.get("PYTHONHASHSEED", "unset"))
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    set_random_seeds(args.seed)

    trainer = LiProSTrainer(args)
    if args.mode == "bo":
        threshold, values = trainer.bayesian_optimization()
        metrics = dict(zip(METRIC_NAMES, map(float, values), strict=True))
        confidence = {"best_threshold": float(threshold)}
    else:
        result = trainer.train(args.confidence_threshold)
        metrics = {name: getattr(result, name) for name in METRIC_NAMES}
        confidence = {"confidence_threshold": args.confidence_threshold}

    if args.epoch_csv:
        fields = (
            "trial", "confidence_threshold", "fold", "phase", "stage", "epoch",
            "loss", "cross_entropy", "consistency_loss", "decision_threshold", *METRIC_NAMES,
        )
        with atomic_output(args.epoch_csv) as temporary, temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trainer.epoch_rows)
        logger.info("[epochs] %s", args.epoch_csv)

    if args.result_json:
        result_path = Path(args.result_json).expanduser()
        payload = {
            "model": "LiProS", "mode": args.mode, "seed": args.seed, "config": vars(args),
            "dataset": trainer.dataset_metadata, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            **confidence, "decision_threshold": trainer.selected_decision_threshold,
            "metrics": metrics, "selected_trial": trainer.selected_trial, "folds": trainer.fold_results,
        }
        with atomic_output(result_path) as temporary, temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str, allow_nan=False)
            handle.write("\n")
        logger.info("[result] %s", result_path)


if __name__ == "__main__":
    raise SystemExit(main(command_line=True))
