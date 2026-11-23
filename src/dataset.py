from pathlib import Path

import torch


DATASET_VERSION = 2
LABELED_SPLITS = ("train", "dev", "test")
UNLABELED = -1
RELATION_NAMES = ("followings", "followers")
CRESCI_RELATION_NAMES = ("follow", "friend")
STATIC_WIDTHS = {"creation": 8, "production": 6, "dissemination": 10,
                 "interactive": 7, "operational": 6}


def load_dataset(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Generate it with python -m src.feature_preprocessing.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") not in (1, DATASET_VERSION):
        raise ValueError("Unsupported LiProS dataset format. Rebuild it with python -m src.feature_preprocessing.")

    labels = data["labels"]
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("Expected a one-dimensional integer label tensor.")
    classes = set(labels.unique().tolist())
    if classes not in ({0, 1}, {UNLABELED, 0, 1}):
        raise ValueError("Expected both binary classes (0 and 1). Only support rows may use -1.")
    n = labels.numel()
    split_sizes = data.get("split_sizes")
    if split_sizes is not None:
        if (not isinstance(split_sizes, dict)
                or set(split_sizes) not in (set(LABELED_SPLITS), {*LABELED_SPLITS, "support"})
                or any(type(size) is not int or size < 1 for size in split_sizes.values())
                or sum(split_sizes.values()) != n):
            raise ValueError("Invalid split sizes in the LiProS dataset.")
        n_labeled = sum(split_sizes[split] for split in LABELED_SPLITS)
        if (labels[:n_labeled].eq(UNLABELED).any()
                or not labels[n_labeled:].eq(UNLABELED).all()):
            raise ValueError("Expected labeled train/dev/test rows followed by unlabeled support rows.")
    elif UNLABELED in classes:
        raise ValueError("A dataset with support rows must include its split sizes.")
    for name in ("description", "tweets", "numeric", "categorical"):
        matrix = data[name]
        if (matrix.ndim != 2 or matrix.size(0) != n or matrix.size(1) == 0
                or not matrix.is_floating_point() or not torch.isfinite(matrix).all()):
            raise ValueError(f"Invalid or misaligned feature matrix: {name}")
    if data["numeric"].size(1) != 6 or data["categorical"].size(1) != 11:
        raise ValueError("Expected 6 numeric and 11 categorical columns.")

    edges, types = data["edge_index"], data["edge_type"]
    if (edges.ndim != 2 or edges.size(0) != 2 or edges.dtype != torch.long
            or types.shape != (edges.size(1),) or types.dtype != torch.long):
        raise ValueError("Invalid edge or relation tensor shape.")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("Edge endpoint is outside the account table.")
    if not set(types.unique().tolist()) <= {0, 1}:
        raise ValueError("Expected edge relation IDs 0 and 1.")

    for name, width in STATIC_WIDTHS.items():
        matrix = data["lifecycle"][name]
        if (matrix.shape != (n, width) or not matrix.is_floating_point()
                or not torch.isfinite(matrix).all()):
            raise ValueError(f"Invalid lifecycle feature matrix: {name}")
    if data["lifecycle"]["interactive"][:, 2:4].count_nonzero().item():
        raise ValueError("Neighbor labels must be computed during training, not stored in the dataset.")
    return data
