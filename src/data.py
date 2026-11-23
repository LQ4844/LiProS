import numpy as np
import torch

from src.dataset import LABELED_SPLITS, UNLABELED


def create_masks(labels, label_rate, rng, folds=1, original_sizes=None):
    """Return one (training, validation, test) mask tuple per fold."""
    y = labels.cpu().numpy()
    labeled = np.flatnonzero(y != UNLABELED)
    mask = np.zeros((folds, 3, y.size), dtype=bool)
    if original_sizes is not None:
        boundaries = np.cumsum([0, *(original_sizes[split] for split in LABELED_SPLITS)])
        train, dev, test = [
            np.arange(start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
        mask[:, 1, dev] = True
        mask[:, 2, test] = True
        for label in (0, 1):
            eligible = train[y[train] == label]
            if not eligible.size:
                raise ValueError("The original training split must contain both binary classes.")
            count = max(1, round(eligible.size * label_rate / 100))
            for fold in range(folds):
                mask[fold, 0, rng.choice(eligible, count, replace=False)] = True
    else:
        n_train = int(labeled.size * label_rate / 100)
        n_dev = int((labeled.size - n_train) * 0.1)
        if n_train < 2 or n_dev < 1:
            raise ValueError(f"Label rate {label_rate:g}% leaves too few training or validation accounts.")
        for fold in range(folds):
            for _ in range(10000):
                perm = rng.permutation(labeled)
                if np.unique(y[perm[:n_train]]).size == 2:
                    break
            else:
                raise ValueError(f"Could not create a binary training fold at {label_rate:g}%.")
            mask[fold, 0, perm[:n_train]] = True
            mask[fold, 1, perm[n_train:n_train + n_dev]] = True
            mask[fold, 2, perm[n_train + n_dev:]] = True
    return torch.from_numpy(mask)


def prepare_training_splits(dataset, args):
    labels = dataset["labels"]
    original_sizes = None
    if getattr(args, "use_original_splits", False):
        original_sizes = dataset.get("split_sizes")
        if original_sizes is None:
            raise ValueError("--use-original-splits requires split metadata. Rebuild the feature file.")
    masks = create_masks(
        labels, args.label_rate, np.random.default_rng(getattr(args, "split_seed", 0)),
        folds=args.folds, original_sizes=original_sizes,
    )
    pseudo_label_scope = torch.ones_like(labels, dtype=torch.bool)
    if original_sizes is not None:
        end = sum(original_sizes[split] for split in LABELED_SPLITS)
        pseudo_label_scope[original_sizes["train"]:end] = False
    return masks, pseudo_label_scope
