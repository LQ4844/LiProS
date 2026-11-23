"""Report mean and sample standard deviation across independent LiProS run seeds."""

import argparse
from glob import glob
import json
from math import isfinite
from pathlib import Path
from statistics import mean, stdev


TEST_METRICS = ("test_acc", "test_precision", "test_recall", "test_f1")
PER_RUN_CONFIG = {"seed", "result_json", "epoch_csv", "checkpoint_dir"}


def aggregate(paths):
    if len(paths) < 2:
        raise ValueError("Provide at least two result JSON files.")
    expected = None
    seeds = set()
    samples = {metric: [] for metric in TEST_METRICS}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        if run.get("model") != "LiProS":
            raise ValueError(f"{path} is not a LiProS training result.")
        seed = run["seed"]
        if type(seed) is not int or seed in seeds:
            raise ValueError(f"{path} must have a distinct integer model seed.")
        config = {name: value for name, value in run["config"].items() if name not in PER_RUN_CONFIG}
        masks = tuple(fold.get("split_sha256") for fold in run.get("folds", ()))
        comparison = config, run.get("dataset"), masks
        if expected is None:
            expected = comparison
        elif expected != comparison:
            raise ValueError(f"{path} has a different configuration, dataset, or split.")
        for metric in TEST_METRICS:
            value = float(run["metrics"][metric])
            if not isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{path} has an invalid percentage for {metric}.")
            samples[metric].append(value)
        seeds.add(seed)
    return {
        "seeds": sorted(seeds),
        "n_runs": len(seeds),
        "metrics": {
            metric: {"mean": mean(values), "sample_stddev": stdev(values)}
            for metric, values in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="+", help="Result JSON paths or glob patterns.")
    args = parser.parse_args()
    paths = []
    for pattern in args.result:
        matches = sorted(glob(pattern))
        if not matches:
            parser.error(f"No result files match {pattern!r}.")
        paths.extend(Path(match) for match in matches)
    try:
        report = aggregate(paths)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
