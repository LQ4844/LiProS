# Metric and training helper provenance is listed under License and acknowledgements in README.md.
from contextlib import contextmanager
import os
import os.path as osp
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
import numpy as np
import torch
from sklearn.metrics import precision_score,recall_score,f1_score


METRIC_NAMES = tuple(
    f"{split}_{metric}"
    for metric in ("acc", "precision", "recall", "f1")
    for split in ("train", "val", "test")
)


@contextmanager
def atomic_output(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{}_{}_".format(name, val)
        st += st_

    return st[:-1]

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def compute_accuracy(preds, labels, train_mask, val_mask, test_mask, positive_index=1):
    
    train_preds = preds[train_mask]
    val_preds = preds[val_mask]
    test_preds = preds[test_mask]
    train_acc = (torch.sum(train_preds == labels[train_mask])).float() / ((labels[train_mask].shape[0]))
    val_acc = (torch.sum(val_preds == labels[val_mask])).float() / ((labels[val_mask].shape[0]))
    test_acc = (torch.sum(test_preds == labels[test_mask])).float() / ((labels[test_mask].shape[0]))

    train_preds=train_preds.to('cpu').detach().numpy()
    val_preds=val_preds.to('cpu').detach().numpy()
    test_preds=test_preds.to('cpu').detach().numpy()
    labels_train=labels[train_mask].to('cpu').detach().numpy()
    labels_val=labels[val_mask].to('cpu').detach().numpy()
    labels_test=labels[test_mask].to('cpu').detach().numpy()
    train_precision=precision_score(labels_train, train_preds, pos_label=positive_index, zero_division=0) *100
    val_precision=precision_score(labels_val, val_preds, pos_label=positive_index, zero_division=0) *100
    test_precision=precision_score(labels_test, test_preds, pos_label=positive_index, zero_division=0) *100
    train_recall=recall_score(labels_train, train_preds, pos_label=positive_index, zero_division=0) *100
    val_recall=recall_score(labels_val, val_preds, pos_label=positive_index, zero_division=0) *100
    test_recall=recall_score(labels_test, test_preds, pos_label=positive_index, zero_division=0) *100
    train_f1=f1_score(labels_train, train_preds, pos_label=positive_index, zero_division=0) *100
    val_f1=f1_score(labels_val, val_preds, pos_label=positive_index, zero_division=0) *100
    test_f1=f1_score(labels_test, test_preds, pos_label=positive_index, zero_division=0) *100
    train_acc = train_acc * 100
    val_acc = val_acc * 100
    test_acc = test_acc * 100

    return train_acc, val_acc, test_acc, train_precision, val_precision,test_precision, train_recall, val_recall, test_recall, train_f1,val_f1, test_f1


def select_binary_threshold(probabilities, labels, lower=0.3, upper=0.9, steps=121, positive_index=1):
    probabilities = probabilities.detach().float().reshape(-1)
    labels = labels.detach().long().reshape(-1)
    if probabilities.numel() != labels.numel() or probabilities.numel() == 0:
        raise ValueError("probabilities and labels must be nonempty vectors of equal length.")
    thresholds = torch.linspace(
        float(lower), float(upper), int(steps), device=probabilities.device
    )
    positive = labels.eq(int(positive_index))
    prediction = probabilities.unsqueeze(0).ge(thresholds.unsqueeze(1))
    tp = (prediction & positive.unsqueeze(0)).sum(dim=1).float()
    fp = (prediction & ~positive.unsqueeze(0)).sum(dim=1).float()
    fn = (~prediction & positive.unsqueeze(0)).sum(dim=1).float()
    tn = (~prediction & ~positive.unsqueeze(0)).sum(dim=1).float()
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)
    accuracy = (tp + tn) / float(labels.numel())
    score = f1 + accuracy
    best_score = score.max()
    tied = torch.nonzero(score.eq(best_score), as_tuple=False).flatten()
    nearest = (thresholds[tied] - 0.5).abs().argmin()
    best_index = tied[nearest]
    return float(thresholds[best_index]), float(f1[best_index] * 100.0), float(accuracy[best_index] * 100.0)


def masking(fold, split_masks, label_rate=0.01):
    key = None
    for rate in split_masks.keys():
        if abs(float(rate) - float(label_rate)) < 1e-9:
            key = rate
            break
    if key is None:
        supported = sorted(split_masks.keys(), key=float)
        raise ValueError(
            "Unsupported label_rate={}. Supported rates: {}.".format(label_rate, supported)
        )
    train_masks, val_masks, test_masks = split_masks[key]
    return train_masks[fold], val_masks[fold], test_masks[fold]

def compute_representation(net, data, device):

    net.eval()
    reps = []

    data = data.to(device)
    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0)

    return reps
