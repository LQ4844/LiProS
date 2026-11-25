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


def resolve_split_mode(dataset, args):
    if getattr(args, "use_original_splits", False):
        return "official"
    mode = getattr(args, "split_mode", "auto")
    if mode != "auto":
        return mode
    if dataset.get("dataset") == "cresci-15" and dataset.get("split_sizes") is not None:
        return "official"
    return "random"


def holdout_support_labels(dataset, args):
    """Hide a fixed set of labels across folds and training seeds."""
    labels = dataset["labels"]
    count = getattr(args, "holdout_support", 0)
    if count == 0:
        return labels
    if resolve_split_mode(dataset, args) != "random":
        raise ValueError("--holdout-support requires random splits.")

    eligible = labels.ne(UNLABELED).nonzero(as_tuple=True)[0].cpu().numpy()
    if not 0 < count < eligible.size:
        raise ValueError("--holdout-support must leave labeled accounts for train, dev, and test.")
    rng = np.random.default_rng([getattr(args, "split_seed", 0), 1])
    support = rng.choice(eligible, size=count, replace=False)
    labels = labels.clone()
    labels[support] = UNLABELED
    if not (labels.eq(0).any() and labels.eq(1).any()):
        raise ValueError("--holdout-support must leave labeled accounts in both classes.")
    return labels


def prepare_training_splits(dataset, args, *, labels=None):
    """Return the split masks and a pseudo-label scope for each fold."""
    original_sizes = None
    if resolve_split_mode(dataset, args) == "official":
        original_sizes = dataset.get("split_sizes")
        if original_sizes is None:
            raise ValueError("--split-mode official requires split metadata. Rebuild the feature file.")
    masks = create_masks(
        dataset["labels"] if labels is None else labels,
        args.label_rate, np.random.default_rng(getattr(args, "split_seed", 0)),
        folds=args.folds, original_sizes=original_sizes,
    )
    if getattr(args, "pseudo_pool", "vu") == "vu":
        pseudo_label_scopes = ~(masks[:, 1] | masks[:, 2])
    else:
        pseudo_label_scopes = torch.ones_like(masks[:, 0])
    return masks, pseudo_label_scopes
