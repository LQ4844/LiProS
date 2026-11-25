# Multi-stage training provenance is listed under Acknowledgements in README.md.
from collections import namedtuple
from pathlib import Path
import gc
import hashlib
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.optim import AdamW
from src.utils import (METRIC_NAMES, atomic_output, compute_accuracy, reset,
                       select_binary_threshold, set_random_seeds)
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.stats import norm
from embedder import Embedder, inference_mode
from src.argument import _ENCODER_DEFAULTS
from src.dataset import UNLABELED
from src.feature_preprocessing import NEIGHBOR_LABEL_COLUMNS, compute_neighbor_label_features
from layers import FourFeatureProjector


logger = logging.getLogger(__name__)
TrainingResult = namedtuple("TrainingResult", ("status", *METRIC_NAMES))


def ensemble_consensus(bot_probabilities, confidence_threshold, variance_threshold):
    if bot_probabilities.ndim != 2 or bot_probabilities.size(0) < 2:
        raise ValueError("bot_probabilities must be an [M, N] tensor with predictions from at least two models.")
    mean_p = bot_probabilities.mean(dim=0)
    ensemble_var = bot_probabilities.var(dim=0, unbiased=False)
    pred = (mean_p >= 0.5).long()
    confidence = torch.maximum(mean_p, 1.0 - mean_p)
    member_pred = (bot_probabilities >= 0.5).long()
    agreement = member_pred.eq(member_pred[0:1]).all(dim=0)
    selected = (
        agreement
        & (confidence >= float(confidence_threshold))
        & (ensemble_var <= float(variance_threshold))
    )
    return mean_p, pred, agreement, confidence, ensemble_var, selected


def evaluate_ensemble_predictions(
    bot_probabilities,
    confidence_threshold,
    variance_threshold,
    unlabeled_mask=None,
    true_labels=None,
    bot_class_index=1,
    member_names=None,
):
    mean_p, pred, agreement, confidence, ensemble_var, selected = ensemble_consensus(
        bot_probabilities, confidence_threshold, variance_threshold
    )
    if unlabeled_mask is not None:
        unlabeled_mask = unlabeled_mask.to(dtype=torch.bool, device=mean_p.device)
        scope = unlabeled_mask
        selected = selected & unlabeled_mask
        agreement_scope = agreement & unlabeled_mask
    else:
        scope = torch.ones_like(selected, dtype=torch.bool)
        agreement_scope = agreement

    bot_idx = int(bot_class_index)
    if bot_idx not in (0, 1):
        raise ValueError("The bot class index must be 0 or 1 for binary predictions.")
    class_pred = torch.full_like(pred, fill_value=1 - bot_idx)
    class_pred[mean_p >= 0.5] = bot_idx

    n_scope = max(int(scope.sum().item()), 1)
    selected_pred = class_pred[selected]
    n_bot = int((selected_pred == bot_idx).sum().item()) if selected.any() else 0
    n_human = int(selected.sum().item()) - n_bot

    member_names = list(member_names) if member_names is not None else [
        "member_{}".format(i) for i in range(bot_probabilities.size(0))
    ]
    member_candidates = {}
    for name, p in zip(member_names, bot_probabilities):
        member_conf = torch.maximum(p, 1.0 - p)
        member_candidates[name] = int(
            (scope & (member_conf >= float(confidence_threshold))).sum().item()
        )

    conf_passed = scope & agreement & (confidence >= float(confidence_threshold))
    var_passed = conf_passed & (ensemble_var <= float(variance_threshold))

    member_pred = bot_probabilities.ge(0.5)
    pairwise_disagreement = {}
    for i in range(bot_probabilities.size(0)):
        for j in range(i + 1, bot_probabilities.size(0)):
            key = "{}_vs_{}".format(member_names[i], member_names[j])
            diff = member_pred[i].ne(member_pred[j]).float()
            if scope.any():
                pairwise_disagreement[key] = float(diff[scope].mean().item())
            else:
                pairwise_disagreement[key] = 0.0

    n_unlabeled = int(scope.sum().item())
    n_selected = int(selected.sum().item())
    coverage = float(n_selected) / float(n_unlabeled) if n_unlabeled else 0.0

    stats = {
        "mean_probability": mean_p,
        "ensemble_variance": ensemble_var,
        "confidence": confidence,
        "agreement": agreement,
        "selected_mask": selected,
        "pred": class_pred,
        "mean_probability_avg": float(mean_p[scope].mean().item()) if scope.any() else 0.0,
        "ensemble_variance_avg": float(ensemble_var[scope].mean().item()) if scope.any() else 0.0,
        "mean_prediction_variance": float(ensemble_var[scope].mean().item()) if scope.any() else 0.0,
        "agreement_ratio": float(agreement_scope.sum().item()) / float(n_scope),
        "agreement_count": int(agreement_scope.sum().item()),
        "confidence_passed_count": int(conf_passed.sum().item()),
        "variance_passed_count": int(var_passed.sum().item()),
        "selected_count": n_selected,
        "pseudo_label_coverage": coverage,
        "class_distribution": {"bot": n_bot, "human": n_human},
        "member_candidate_counts": member_candidates,
        "pairwise_disagreement": pairwise_disagreement,
    }

    if true_labels is not None:
        true_labels = true_labels.to(device=class_pred.device)
        evaluated = selected & true_labels.ne(UNLABELED)
    if true_labels is not None and evaluated.any():
        y_true = true_labels[evaluated]
        y_hat = class_pred[evaluated]
        correct = y_true.eq(y_hat)
        stats["pseudo_label_accuracy"] = float(correct.float().mean().item())
        tp = int(((y_hat == bot_idx) & (y_true == bot_idx)).sum().item())
        fp = int(((y_hat == bot_idx) & (y_true != bot_idx)).sum().item())
        fn = int(((y_hat != bot_idx) & (y_true == bot_idx)).sum().item())
        stats["pseudo_label_precision"] = float(tp) / float(tp + fp) if (tp + fp) else 0.0
        stats["pseudo_label_recall"] = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    return stats


class LiProSTrainer(Embedder):
    def __init__(self, args):
        super().__init__(args)
        self.epoch_rows = []
        self.trial_number = 1
        self.selected_trial = 1
        self.reset_run_state()

    def reset_run_state(self):
        self.fold_results = []
        self.selected_decision_threshold = 0.5

    def reset_fold_state(self):
        self.epochs_without_improvement = 0
        self.best_val_acc = float("-inf")
        self.best_epoch = None

    def evaluate(self, epoch, loss, ce, consistency_loss):
        inputs = (self.des, self.tweet, self.num, self.cat, self.edge_index, self.edge_type)
        bot_index = int(getattr(self.args, "bot_class_index", 1))
        decision_threshold = 0.5
        if getattr(self.args, "calibrate_decision_threshold", False):
            bot_probability = self._evaluation_bot_probabilities(*inputs)
            decision_threshold, _, _ = select_binary_threshold(
                bot_probability[self.val_mask],
                self.y[self.val_mask],
                lower=self.args.decision_threshold_lower,
                upper=self.args.decision_threshold_upper,
                steps=self.args.decision_threshold_steps,
                positive_index=bot_index,
            )
            preds = torch.full_like(self.y, fill_value=1 - bot_index)
            preds[bot_probability >= decision_threshold] = bot_index
        else:
            preds = self._evaluation_predictions(*inputs)

        values = compute_accuracy(preds, self.y, self.train_mask, self.val_mask, self.test_mask, bot_index)
        metrics = dict(zip(METRIC_NAMES, map(float, values), strict=True))
        row = {
            "epoch": epoch, "loss": loss, "cross_entropy": ce, "consistency_loss": consistency_loss,
            "decision_threshold": decision_threshold, **metrics,
        }
        if metrics["val_acc"] > self.best_val_acc:
            self.best_val_acc = metrics["val_acc"]
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        score = metrics["val_f1"] + metrics["val_acc"]
        improved = self.best_epoch is None or score > self.best_epoch["selection_score"]
        if improved:
            self.best_epoch = {**row, "selection_score": score}
        return row, improved

    def save_results(self, fold):
        masks = self.split_masks[fold]
        split_hash = hashlib.sha256(masks.contiguous().numpy().tobytes()).hexdigest()
        split_counts = dict(zip(("train", "dev", "test"), masks.sum(dim=1).tolist(), strict=True))
        self.fold_results.append({
            "fold": fold + 1,
            "split_sha256": split_hash,
            "visible_split_counts": split_counts,
            **self.best_epoch,
        })
        self.selected_decision_threshold = self.best_epoch["decision_threshold"]
        logger.info(
            "[fold %d] selected epoch %d by val F1 + accuracy %.2f | test accuracy %.2f | test F1 %.2f",
            fold + 1, self.best_epoch["epoch"], self.best_epoch["selection_score"],
            self.best_epoch["test_acc"], self.best_epoch["test_f1"],
        )

    def summary(self):
        metrics = tuple(float(np.mean([fold[name] for fold in self.fold_results])) for name in METRIC_NAMES)
        logger.info("[summary] %s", " | ".join(
            f"{name} {value:.2f}" for name, value in zip(METRIC_NAMES, metrics, strict=True)
        ))
        return metrics

    def _unlabeled_candidates(self):
        return self.pseudo_label_scope & ~self.running_train_mask

    def _checkpoint_state(self):
        def cpu_weights(module):
            return {name: tensor.detach().to("cpu", copy=True) for name, tensor in module.state_dict().items()}

        models = self.models if self._ensemble_enabled() else [self.model]
        state = {"models": [cpu_weights(model) for model in models]}
        projector = getattr(self, "shared_input_projector", None)
        if projector is not None:
            state["shared_input_projector"] = cpu_weights(projector)
        return state

    def _write_checkpoint(self, path, weights):
        with atomic_output(path) as temporary:
            torch.save({
                **weights, "trial": self.trial_number, "fold": len(self.fold_results),
                "epoch": self.best_epoch["epoch"],
                "decision_threshold": self.best_epoch["decision_threshold"],
            }, temporary)

    def _clear_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensemble_enabled(self):
        return getattr(self.args, "uncertainty_method", "legacy") == "ensemble"

    def _bot_class_index(self):
        idx = int(getattr(self.args, "bot_class_index", 1))
        if idx < 0 or idx >= self.num_classes:
            raise ValueError(
                "bot_class_index={} is out of range for {} classes.".format(idx, self.num_classes)
            )
        return idx

    def _bot_probabilities_from_logits(self, logits):
        return F.softmax(logits, dim=-1)[:, self._bot_class_index()]

    def _pred_from_mean_bot_prob(self, mean_p):
        bot_idx = self._bot_class_index()
        preds = torch.full_like(mean_p, fill_value=1 - bot_idx, dtype=torch.long)
        preds[mean_p >= 0.5] = bot_idx
        return preds

    def _normalized_ensemble_type(self):
        raw = getattr(self.args, "ensemble_type", "architecture")
        aliases = {
            "architecture": "architecture",
            "gnn_hetero": "architecture",
            "seed": "seed",
            "same_init": "same_init",
            "existing_hetero": "existing_hetero",
        }
        return aliases.get(raw, raw)

    def _ensemble_member_names(self):
        ensemble_type = self._normalized_ensemble_type()
        if ensemble_type == "architecture":
            return ["gcn", "gat", "sargcn"]
        if ensemble_type == "seed":
            return ["sargcn"] * len(self._ensemble_member_seeds())
        if ensemble_type == "same_init":
            return [self.args.encoder] * int(getattr(self.args, "ensemble_size", 3))
        if ensemble_type == "existing_hetero":
            return ["lipros", "sargcn", "dual"]
        raise ValueError("Unknown ensemble_type: {}".format(ensemble_type))

    def _ensemble_member_seeds(self):
        ensemble_type = self._normalized_ensemble_type()
        base_seed = int(getattr(self, "_ensemble_seed_base", 0))
        if ensemble_type == "seed":
            return [int(s) for s in getattr(self.args, "ensemble_seeds", [1, 2, 3])]
        if ensemble_type == "architecture":
            return [base_seed] * 3
        if ensemble_type == "existing_hetero":
            return [base_seed] * 3
        offsets = list(getattr(self.args, "ensemble_seed_offsets", [0, 1000, 2000]))
        size = int(getattr(self.args, "ensemble_size", 3))
        return [base_seed + offsets[i] for i in range(size)]

    def _ensemble_member_labels(self):
        ensemble_type = self._normalized_ensemble_type()
        names = self._ensemble_member_names()
        if ensemble_type == "seed":
            return ["seed{}".format(s) for s in self._ensemble_member_seeds()]
        return list(names)

    def _create_model(self, encoder_name):
        shared = None
        if (
            getattr(self, "shared_input_projector", None) is not None
            and encoder_name in ("gcn", "gat", "sargcn")
        ):
            shared = self.shared_input_projector
        encoder, classifier = self._build_encoder(
            encoder_name, shared_projector=shared
        )
        return LiProS(
            encoder,
            classifier,
            encoder_needs_new_feature=encoder_name in ("lipros", "dual", "esa_cons"),
            dual_space=encoder_name == "dual",
            consist_space=encoder_name == "esa_cons",
        ).to(self.device)

    def _optimizer_config_for(self, encoder_name):
        ensemble_type = self._normalized_ensemble_type() if self._ensemble_enabled() else None
        if ensemble_type in ("architecture", "seed"):
            return (
                float(self.args.lr),
                float(self.args.decay),
                getattr(self.args, "optimizer", _ENCODER_DEFAULTS["sargcn"]["optimizer"]),
            )
        defaults = _ENCODER_DEFAULTS[encoder_name]
        if ensemble_type == "existing_hetero":
            return float(defaults["lr"]), float(defaults["decay"]), defaults["optimizer"]
        return (
            float(self.args.lr),
            float(self.args.decay),
            getattr(self.args, "optimizer", defaults["optimizer"]),
        )

    def _make_optimizer(self, model, encoder_name=None):
        encoder_name = encoder_name or self.args.encoder
        lr, decay, optim_name = self._optimizer_config_for(encoder_name)
        optim_cls = AdamW if optim_name == "adamw" else Adam
        return optim_cls(model.parameters(), lr=lr, weight_decay=decay), (lr, decay, optim_name)

    def _needs_shared_input_projector(self):
        return self._ensemble_enabled() and self._normalized_ensemble_type() in (
            "architecture",
            "seed",
        )

    def _init_shared_input_projector(self, base_seed):
        self.shared_input_projector = None
        self.shared_projector_optimizer = None
        if not self._needs_shared_input_projector():
            return
        set_random_seeds(base_seed)
        self.shared_input_projector = FourFeatureProjector(
            des_size=self.des.size(1),
            tweet_size=self.tweet.size(1),
            num_prop_size=self.num.size(1),
            cat_prop_size=self.cat.size(1),
            embedding_dimension=self.rep_size,
        ).to(self.device)
        lr, decay, optim_name = self._optimizer_config_for("sargcn")
        optim_cls = AdamW if optim_name == "adamw" else Adam
        self.shared_projector_optimizer = optim_cls(
            self.shared_input_projector.parameters(), lr=lr, weight_decay=decay
        )
        logger.info(
            "[ensemble] shared_input_embedding=FourFeatureProjector dim={} | "
            "type={} | Graph models use independent parameters and state dictionaries".format(
                self.rep_size, self._normalized_ensemble_type()
            )
        )

    def _init_model(self):
        member_names = (
            self._ensemble_member_names() if self._ensemble_enabled() else [self.args.encoder]
        )
        member_labels = (
            self._ensemble_member_labels() if self._ensemble_enabled() else [self.args.encoder]
        )
        member_seeds = (
            self._ensemble_member_seeds()
            if self._ensemble_enabled()
            else [int(getattr(self, "_ensemble_seed_base", 0))]
        )
        base_seed = int(getattr(self, "_ensemble_seed_base", 0))
        self._init_shared_input_projector(base_seed)

        self.models = []
        self.optimizers = []
        optim_specs = []
        for encoder_name, seed in zip(member_names, member_seeds):
            set_random_seeds(seed)
            model = self._create_model(encoder_name)
            optimizer, spec = self._make_optimizer(model, encoder_name)
            self.models.append(model)
            self.optimizers.append(optimizer)
            optim_specs.append(spec)

        if len(self.models) >= 2:
            ids0 = {id(p) for p in self.models[0].parameters()}
            for other in self.models[1:]:
                ids_other = {id(p) for p in other.parameters()}
                if not ids0.isdisjoint(ids_other):
                    raise RuntimeError("Ensemble initialization failed because members share parameters.")

        self.model = self.models[0]
        self.optimizer = self.optimizers[0]
        self.encoder = self.model.encoder
        self.classifier = self.model.classifier
        primary_name = member_names[0]
        self.encoder_needs_new_feature = primary_name in ("lipros", "dual", "esa_cons")
        self.dual_space = primary_name == "dual"
        self.consist_space = primary_name == "esa_cons"
        self._ensemble_member_names_cache = member_names
        self._ensemble_member_labels_cache = member_labels

        self.attn_filter_model = None
        self.attn_filter_optimizer = None
        if getattr(self.args, "use_attention_filter", False) and not any(
            self._model_has_attention(model) for model in self.models
        ):
            set_random_seeds(base_seed + 9999)
            self.attn_filter_model = self._create_model("lipros")
            self.attn_filter_optimizer, attn_spec = self._make_optimizer(
                self.attn_filter_model, "lipros"
            )
            logger.info(
                "[attn-filter] Ensemble members lack ESA attention weights. Training a separate ESA model to compute attention scores | "
                "lr={} decay={} | Excluded from pseudo-label voting and evaluation".format(
                    attn_spec[0], attn_spec[1]
                )
            )

        if self._ensemble_enabled():
            ens_type = self._normalized_ensemble_type()
            kind = {
                "architecture": "Ensemble-A (architecture diversity)",
                "seed": "Ensemble-B (same-architecture seed/deep ensemble)",
                "same_init": "same-architecture seed ensemble",
                "existing_hetero": "existing hetero ensemble",
            }.get(ens_type, ens_type)
            optim_desc = "; ".join(
                "{}:{} lr={} decay={}".format(label, opt, lr, decay)
                for label, (lr, decay, opt) in zip(member_labels, optim_specs)
            )
            logger.info(
                "[ensemble] {} | members={} | seeds={} | reinit_each_stage={} | {}".format(
                    kind,
                    ",".join(member_labels),
                    ",".join(str(s) for s in member_seeds),
                    getattr(self.args, "ensemble_reinit_each_stage", False),
                    optim_desc,
                )
            )

    def _maybe_init_ensemble_models(self, stage):
        reinit = bool(getattr(self.args, "ensemble_reinit_each_stage", False))
        need = (
            (not hasattr(self, "models"))
            or (self.models is None)
            or (len(getattr(self, "models", [])) == 0)
            or reinit
            or (stage == 0)
        )
        if need:
            self._init_model()
        else:
            logger.info(
                "[ensemble] stage={} Continuing training from the previous stage's model parameters".format(
                    stage + 1
                )
            )

    def _logits(self, edge_index, edge_type):
        return self._logits_for(self.model, edge_index, edge_type)

    def _logits_for(self, model, edge_index, edge_type):
        return model.cls(
            self.des, self.tweet, self.num, self.cat,
            edge_index, edge_type, new_feature=self.new_feature,
        )[0]

    def _representation(self):
        return self.model.represent(
            self.des, self.tweet, self.num, self.cat,
            self.edge_index, self.edge_type, new_feature=self.new_feature,
        )

    def _l2_penalty(self, model):
        coefficient = getattr(self.args, "l2_lambda", 1e-5)
        if coefficient == 0:
            return None
        return coefficient * sum(
            parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad
        )

    def _loss_for_model(self, model, edge_index, edge_type):
        cons = None
        if getattr(model, "consist_space", False):
            logits, z_esa, z_graph = model.forward_with_reps(
                self.des, self.tweet, self.num, self.cat,
                edge_index, edge_type, new_feature=self.new_feature,
            )
            logits_cnn, z_esa_cnn, z_graph_cnn = model.forward_with_reps(
                self.des, self.tweet, self.num, self.cat,
                self.cnn_edge_index, self.cnn_edge_type, new_feature=self.new_feature,
            )
            cons = 0.5 * (
                F.mse_loss(F.normalize(z_esa, dim=1), F.normalize(z_graph, dim=1))
                + F.mse_loss(F.normalize(z_esa_cnn, dim=1), F.normalize(z_graph_cnn, dim=1))
            )
        else:
            logits = self._logits_for(model, edge_index, edge_type)
            logits_cnn = self._logits_for(model, self.cnn_edge_index, self.cnn_edge_type)
        mask = self.running_train_mask
        targets = self.labels[mask]
        ce = F.cross_entropy(logits[mask], targets) + F.cross_entropy(logits_cnn[mask], targets)
        loss = ce
        if cons is not None:
            loss = loss + float(getattr(self.args, "consist_lambda", 0.1)) * cons
        penalty = self._l2_penalty(model)
        if penalty is not None:
            loss = loss + penalty
        return loss, ce.detach(), None if cons is None else cons.detach()

    def _supervised_loss(self, edge_index, edge_type):
        return self._loss_for_model(self.model, edge_index, edge_type)

    def _train_models_step(self, edge_index, edge_type):
        if not self._ensemble_enabled():
            self.model.train()
            self.optimizer.zero_grad()
            loss, ce_det, cons_det = self._supervised_loss(edge_index, edge_type)
            loss.backward()
            self.optimizer.step()
            return float(loss.detach()), float(ce_det), None if cons_det is None else float(cons_det)

        if getattr(self, "shared_projector_optimizer", None) is not None:
            self.shared_input_projector.train()
            self.shared_projector_optimizer.zero_grad()

        losses, ces, consistencies = [], [], []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
            loss, ce_det, cons_det = self._loss_for_model(model, edge_index, edge_type)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            ces.append(float(ce_det))
            if cons_det is not None:
                consistencies.append(float(cons_det))

        shared_regularization = 0.0
        if getattr(self, "shared_projector_optimizer", None) is not None:
            # Ensemble members keep the shared projector outside their parameter lists.
            penalty = self._l2_penalty(self.shared_input_projector)
            if penalty is not None:
                penalty.backward()
                shared_regularization = float(penalty.detach())
            self.shared_projector_optimizer.step()

        if getattr(self, "attn_filter_model", None) is not None:
            self.attn_filter_model.train()
            self.attn_filter_optimizer.zero_grad()
            attn_loss, _, _ = self._loss_for_model(
                self.attn_filter_model, edge_index, edge_type
            )
            attn_loss.backward()
            self.attn_filter_optimizer.step()
        return (
            (sum(losses) + shared_regularization) / len(losses),
            float(np.mean(ces)),
            float(np.mean(consistencies)) if consistencies else None,
        )

    def _evaluation_predictions(self, des, tweet, num, cat, edge_index, edge_type):
        if not self._ensemble_enabled():
            return super()._evaluation_predictions(
                des, tweet, num, cat, edge_index, edge_type
            )
        return self._pred_from_mean_bot_prob(self._collect_bot_probabilities().mean(dim=0))

    def _evaluation_bot_probabilities(self, *inputs):
        if self._ensemble_enabled():
            return self._collect_bot_probabilities().mean(dim=0)
        return super()._evaluation_bot_probabilities(*inputs)

    def _model_has_attention(self, model):
        return bool(
            getattr(model, "encoder_needs_new_feature", False)
            or getattr(model, "dual_space", False)
            or getattr(model, "consist_space", False)
        )

    def _attention_source_model(self):
        if getattr(self, "attn_filter_model", None) is not None:
            return self.attn_filter_model
        for model in getattr(self, "models", []):
            if self._model_has_attention(model):
                return model
        if self._model_has_attention(self.model):
            return self.model
        return None

    def _compute_attention_score(self):
        model = self._attention_source_model()
        if model is None:
            return None
        with inference_mode(model):
            if getattr(model, "dual_space", False):
                model.encoder.forward_feature(
                    self.des, self.tweet, self.num, self.cat,
                    self.new_feature, self.edge_index, self.edge_type,
                )
            elif getattr(model, "consist_space", False):
                model.encoder.forward_esa(
                    self.des, self.tweet, self.num, self.cat,
                    self.new_feature, self.edge_index, self.edge_type,
                )
            else:
                model.encoder(
                    self.des, self.tweet, self.num, self.cat,
                    self.new_feature, self.edge_index, self.edge_type,
                )
            alpha = model.encoder.get_node_attention()
        eps = 1e-12
        k = alpha.size(1)
        entropy = -(alpha * (alpha + eps).log()).sum(dim=1)
        h_norm = entropy / float(np.log(k))
        return (1.0 - h_norm).clamp(0.0, 1.0)

    def _attn_filter_enabled(self):
        if not getattr(self.args, "use_attention_filter", False):
            return False
        if self._attention_source_model() is None:
            logger.info("[attn-filter] Skipping the attention score filter because ESA attention weights are unavailable")
            return False
        return True


    def _init_dataset(self):
        self.running_train_mask = self.train_mask.clone()
        self.labels = torch.zeros_like(self.y)
        self.labels[self.train_mask] = self.y[self.train_mask]

    def _cluster_pseudo_labels(self):
        with inference_mode(self.model):
            representations = F.normalize(self._representation(), dim=1).cpu().numpy()
        candidates = self._unlabeled_candidates().cpu().numpy()
        clustering = KMeans(n_clusters=self.args.num_K, n_init=10).fit(representations[candidates])
        classes = self.labels.cpu().numpy()
        known = self.running_train_mask.cpu().numpy()
        centroids = np.stack([
            representations[known & (classes == label)].mean(axis=0)
            for label in range(self.num_classes)
        ])
        distances = np.square(clustering.cluster_centers_[:, None] - centroids).sum(axis=2)
        cluster_classes = distances.argmin(axis=1)
        pseudo_labels = np.full(representations.shape[0], -1, dtype=np.int64)
        pseudo_labels[candidates] = cluster_classes[clustering.labels_]
        return pseudo_labels

    def pretrain(self, mask, stage,confidece_threshold):
        self._current_stage = stage
        pretrain_epochs = getattr(self.args, "pretrain_epochs", 200)
        for epoch in range(pretrain_epochs):
            loss, ce_det, cons_det = self._train_models_step(
                self.edge_index, self.edge_type
            )

            if cons_det is not None:
                st = '[Fold : {}][Stage : {}/{}][Epoch {}/{}] Loss: {:.4f} | CE: {:.4f} | Cons: {:.4f}'.format(
                    mask+1, stage+1, self.args.stage, epoch+1, pretrain_epochs,
                    loss, ce_det, cons_det,
                )
            else:
                st = '[Fold : {}][Stage : {}/{}][Epoch {}/{}] Loss: {:.4f}'.format(
                    mask+1, stage+1, self.args.stage, epoch+1, pretrain_epochs, loss
                )
            logger.info(st)
            if getattr(self.args, "epoch_csv", None):
                self.epoch_rows.append({
                    "trial": self.trial_number, "confidence_threshold": confidece_threshold,
                    "fold": mask + 1, "phase": "pretrain", "stage": stage + 1, "epoch": epoch + 1,
                    "loss": loss, "cross_entropy": ce_det, "consistency_loss": cons_det,
                })

        if self.args.clustering:
            pseudo_labels = self._cluster_pseudo_labels()
            y_train, self.running_train_mask = self.selftraining_with_checking(pseudo_labels,confidece_threshold)
        else:
            y_train, self.running_train_mask = self.selftraining(confidece_threshold)

        self.labels[self.running_train_mask] = torch.argmax(y_train[self.running_train_mask], dim=1)
        self._clear_cuda()

    def train(self,confidece_threshold):

        self.reset_run_state()
        checkpoint_dir = getattr(self.args, "checkpoint_dir", None)
        write_epochs = bool(getattr(self.args, "epoch_csv", None))
        status = ""
        min_candidates = max(getattr(self.args, "min_unlabeled", 1),
                             self.args.num_K if self.args.clustering else 1)
        logger.info("[run] confidence_threshold = {:.6f}".format(float(confidece_threshold)))
        if getattr(self.args, "use_attention_filter", False):
            logger.info(
                "[run] attention_filter ON | γ={:.4f} | Attention Score = 1-H_norm(α)".format(
                    float(getattr(self.args, "attention_threshold", 0.3))
                )
            )
        if getattr(self, "consist_space", False):
            logger.info(
                "[run] consist_space ON | λ={:.4f} | L=CE+λ·MSE(norm(z_esa),norm(z_graph))".format(
                    float(getattr(self.args, "consist_lambda", 0.1))
                )
            )
        for fold in range(self.args.folds):
            self.reset_fold_state()
            weights = None
            run_seed = int(getattr(self.args, "seed", 0))
            fold_seed = run_seed + fold
            set_random_seeds(fold_seed)
            self.train_mask, self.val_mask, self.test_mask = self.split_masks[fold].to(self.device)
            self.pseudo_label_scope = self.pseudo_label_scopes[fold]
            neighbor_features = compute_neighbor_label_features(self.edge_index, self.y, self.train_mask)
            self.new_feature[:, NEIGHBOR_LABEL_COLUMNS] = neighbor_features.to(self.device)
            logger.info(f"[features] Fold {fold + 1} uses only {int(self.train_mask.sum().item())} "
                  "training nodes for neighbor label features")
            self._init_dataset()

            stage_seed_base = run_seed * 1000000 + fold * 100000
            self._ensemble_seed_base = stage_seed_base
            self._init_model()
            for stage in range(self.args.stage):
                candidates = int(self._unlabeled_candidates().sum().item())
                if candidates < min_candidates:
                    logger.info("[self-training] Fold %d stopped after %d rounds: %d candidates, minimum %d.",
                                fold + 1, stage, candidates, min_candidates)
                    break
                if stage > 0:
                    self._ensemble_seed_base = stage_seed_base + stage * 10000
                    self._maybe_init_ensemble_models(stage)
                self.pretrain(fold, stage, confidece_threshold)
                self._clear_cuda()

            if getattr(self.args, "reinit_final", False):
                self._ensemble_seed_base = stage_seed_base + self.args.stage * 10000
                self._init_model()
                logger.info("[run] reinit_final: Training from scratch on the final labeled set of {} nodes".format(
                    int(self.running_train_mask.sum().item())
                ))
                
            for epoch in range(1,self.args.epochs+1):
                loss, ce_det, cons_det = self._train_models_step(
                    self.edge_index, self.edge_type
                )
                row, improved = self.evaluate(epoch, loss, ce_det, cons_det)
                if write_epochs:
                    self.epoch_rows.append({
                        "trial": self.trial_number, "confidence_threshold": confidece_threshold,
                        "fold": fold + 1, "phase": "final", **row,
                    })
                if checkpoint_dir and improved:
                    weights = self._checkpoint_state()
                status = (f"[fold {fold + 1}][epoch {epoch}/{self.args.epochs}] loss {loss:.4f} | "
                          f"val accuracy {row['val_acc']:.2f} | val F1 {row['val_f1']:.2f}")
                logger.info(status)
                if self.epochs_without_improvement >= self.args.patience:
                    logger.info("Early stopping after %d epochs without a higher validation accuracy.",
                                self.args.patience)
                    break
            self.save_results(fold)
            if checkpoint_dir:
                path = Path(checkpoint_dir).expanduser() / f"trial-{self.trial_number:02d}-fold-{fold + 1:02d}.pt"
                self._write_checkpoint(path, weights)
                self.fold_results[-1]["checkpoint"] = str(path)
            self._clear_cuda()
        return TrainingResult(status, *self.summary())
        
        
    def fitness_function(self,confidece_threshold):
        result = self.train(confidece_threshold)
        return torch.tensor([getattr(result, name) for name in METRIC_NAMES], dtype=torch.float64)
    
    def bayesian_optimization(self, initial_points=None, iterations=None):
        lower = float(getattr(self.args, "bo_lower", 0.5))
        upper = float(getattr(self.args, "bo_upper", 0.9))
        initial_points = (
            initial_points
            if initial_points is not None
            else int(getattr(self.args, "bo_initial_points", 3))
        )
        iterations = (
            iterations
            if iterations is not None
            else int(getattr(self.args, "bo_iterations", 7))
        )
        candidate_count = int(getattr(self.args, "bo_candidates", 512))
        xi = float(getattr(self.args, "bo_xi", 0.01))
        metric_name = getattr(self.args, "bo_metric", "val_f1")
        metric_index = METRIC_NAMES.index(metric_name)
        rng = np.random.RandomState(int(getattr(self.args, "bo_seed", 0)))

        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("Bayesian optimization bounds must satisfy 0 <= lower < upper <= 1.")
        if initial_points < 2 or iterations < 0 or candidate_count < 2:
            raise ValueError("Require bo_initial_points >= 2, bo_iterations >= 0, and bo_candidates >= 2.")

        thresholds = []
        metric_history = []
        fold_history = []

        def evaluate_threshold(threshold):
            threshold = float(threshold)
            self.trial_number = len(thresholds) + 1
            logger.info(
                "[BO] Evaluation {}/{} | threshold={:.6f}".format(
                    len(thresholds) + 1, initial_points + iterations, threshold
                )
            )
            metrics = self.fitness_function(threshold).detach().cpu().numpy().astype(float)
            thresholds.append(threshold)
            metric_history.append(metrics)
            fold_history.append(self.fold_results)
            logger.info("[BO] {}={:.6f}".format(metric_name, metrics[metric_index]))

        initial = [lower, upper]
        if initial_points > 2:
            initial.extend(rng.uniform(lower, upper, size=initial_points - 2).tolist())
        for threshold in initial[:initial_points]:
            evaluate_threshold(threshold)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=0.1, length_scale_bounds=(1e-3, 1.0), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
        )
        for _ in range(iterations):
            x_history = np.asarray(thresholds, dtype=float).reshape(-1, 1)
            y_history = np.asarray(metric_history, dtype=float)[:, metric_index]
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=rng,
            )
            gp.fit(x_history, y_history)

            candidates = rng.uniform(lower, upper, size=(candidate_count, 1))
            mu, sigma = gp.predict(candidates, return_std=True)
            improvement = mu - np.max(y_history) - xi
            z = np.divide(
                improvement,
                sigma,
                out=np.zeros_like(improvement),
                where=sigma > 1e-12,
            )
            expected_improvement = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            expected_improvement[sigma <= 1e-12] = 0.0
            order = np.argsort(expected_improvement)[::-1]
            next_threshold = None
            for candidate_index in order:
                candidate = float(candidates[candidate_index, 0])
                if all(abs(candidate - old) > 1e-6 for old in thresholds):
                    next_threshold = candidate
                    break
            if next_threshold is None:
                raise RuntimeError("Bayesian optimization could not find an unevaluated threshold.")
            evaluate_threshold(next_threshold)

        scores = np.asarray(metric_history, dtype=float)[:, metric_index]
        best_index = int(np.argmax(scores))
        best_threshold = thresholds[best_index]
        best_metrics = metric_history[best_index]
        self.selected_trial = best_index + 1
        self.fold_results = fold_history[best_index]
        self.selected_decision_threshold = self.fold_results[-1]["decision_threshold"]
        logger.info("[BO] Best confidence threshold: {:.6f}".format(best_threshold))
        logger.info("[BO] Selection metric: {}={:.6f}".format(metric_name, best_metrics[metric_index]))
        logger.info(
            "[BO] Metrics at the selected threshold: {}".format(
                ", ".join(
                    "{}={:.6f}".format(name, value)
                    for name, value in zip(METRIC_NAMES, best_metrics, strict=True)
                )
            )
        )
        return best_threshold, best_metrics

    def _collect_bot_probabilities(self):
        bot_probs = []
        for model in self.models:
            with inference_mode(model):
                logits = self._logits_for(model, self.edge_index, self.edge_type)
                bot_probs.append(self._bot_probabilities_from_logits(logits))
        return torch.stack(bot_probs)

    def evaluate_ensemble_predictions(self, confidence_threshold=None, variance_threshold=None):
        if not self._ensemble_enabled():
            raise RuntimeError("evaluate_ensemble_predictions requires ensemble mode.")
        confidence_threshold = (
            self.args.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        variance_threshold = float(
            getattr(self.args, "ensemble_variance_threshold", 0.01)
            if variance_threshold is None
            else variance_threshold
        )
        bot_probabilities = self._collect_bot_probabilities()
        member_names = getattr(
            self,
            "_ensemble_member_labels_cache",
            getattr(self, "_ensemble_member_names_cache", self._ensemble_member_labels()),
        )
        unlabeled_mask = self._unlabeled_candidates()
        stats = evaluate_ensemble_predictions(
            bot_probabilities,
            confidence_threshold=confidence_threshold,
            variance_threshold=variance_threshold,
            unlabeled_mask=unlabeled_mask,
            true_labels=self.y,
            bot_class_index=self._bot_class_index(),
            member_names=member_names,
        )
        logger.info(
            "[evaluate_ensemble] mean_p={:.4f} var={:.6f} agreement_ratio={:.4f} "
            "selected={} coverage={:.4f} class_dist={} pairwise={}".format(
                stats["mean_probability_avg"],
                stats["mean_prediction_variance"],
                stats["agreement_ratio"],
                stats["selected_count"],
                stats["pseudo_label_coverage"],
                stats["class_distribution"],
                stats["pairwise_disagreement"],
            )
        )
        if "pseudo_label_precision" in stats:
            logger.info(
                "[evaluate_ensemble][offline] precision={:.4f} recall={:.4f} accuracy={:.4f}".format(
                    stats["pseudo_label_precision"],
                    stats["pseudo_label_recall"],
                    stats["pseudo_label_accuracy"],
                )
            )
        return stats

    def _log_stage_ensemble_stats(
        self,
        stage,
        member_names,
        member_candidate_counts,
        agreement_count,
        confidence_passed_count,
        variance_passed_count,
        final_selected_count,
        bot_count,
        human_count,
        diversity_stats=None,
        offline_metrics=None,
    ):
        logger.info("Stage {}".format(stage + 1))
        logger.info("-------------------------")
        ens_type = self._normalized_ensemble_type() if self._ensemble_enabled() else ""
        for name in member_names:
            label = {
                "gcn": "GCN",
                "gat": "GAT",
                "sargcn": "SARGCN",
                "lipros": "LiProS",
                "dual": "Dual",
            }.get(name, name)
            if name.startswith("seed"):
                label = "SARGCN({})".format(name)
            logger.info(
                "{} candidate count: {}".format(
                    label, member_candidate_counts.get(name, 0)
                )
            )
        logger.info("Ensemble agreement count: {}".format(agreement_count))
        logger.info("Confidence passed count: {}".format(confidence_passed_count))
        logger.info("Variance passed count: {}".format(variance_passed_count))
        logger.info("Final ensemble selected count: {}".format(final_selected_count))
        logger.info("Bot pseudo-label count: {}".format(bot_count))
        logger.info("Human pseudo-label count: {}".format(human_count))
        if diversity_stats:
            logger.info(
                "mean prediction variance: {:.8f}".format(
                    diversity_stats.get("mean_prediction_variance", 0.0)
                )
            )
            logger.info(
                "pseudo-label coverage: {:.4f}".format(
                    diversity_stats.get("pseudo_label_coverage", 0.0)
                )
            )
            for pair, rate in diversity_stats.get("pairwise_disagreement", {}).items():
                logger.info("prediction disagreement [{}]: {:.4f}".format(pair, rate))
            if ens_type == "seed":
                logger.info(
                    "note: Ensemble-B disagreement reflects initialization/"
                    "dropout/optimization randomness (same-architecture seed ensemble)"
                )
        if offline_metrics:
            logger.info(
                "pseudo-label precision: {:.4f}".format(
                    offline_metrics["pseudo_label_precision"]
                )
            )
            logger.info(
                "pseudo-label recall: {:.4f}".format(
                    offline_metrics["pseudo_label_recall"]
                )
            )

    def _ensemble_selftraining(self, confidece_threshold, pseudo_labels=None):
        bot_probabilities = self._collect_bot_probabilities()
        variance_threshold = float(
            getattr(self.args, "ensemble_variance_threshold", 0.01)
        )
        member_labels = getattr(
            self,
            "_ensemble_member_labels_cache",
            getattr(self, "_ensemble_member_names_cache", self._ensemble_member_labels()),
        )
        unlabeled_mask = self._unlabeled_candidates()

        eval_stats = evaluate_ensemble_predictions(
            bot_probabilities,
            confidence_threshold=confidece_threshold,
            variance_threshold=variance_threshold,
            unlabeled_mask=unlabeled_mask,
            true_labels=self.y,
            bot_class_index=self._bot_class_index(),
            member_names=member_labels,
        )

        mean_p = eval_stats["mean_probability"]
        agreement = eval_stats["agreement"]
        confidence = eval_stats["confidence"]
        ensemble_var = eval_stats["ensemble_variance"]
        ensemble_selected_mask = eval_stats["selected_mask"]
        consensus_prediction = self._pred_from_mean_bot_prob(mean_p)

        agreement_u = unlabeled_mask & agreement
        conf_passed = agreement_u & (confidence >= float(confidece_threshold))
        var_passed = conf_passed & (ensemble_var <= variance_threshold)

        indicator = ensemble_selected_mask.clone()
        if self._attn_filter_enabled():
            attn_score = self._compute_attention_score()
            indicator = indicator & (
                attn_score >= float(getattr(self.args, "attention_threshold", 0.3))
            )

        if pseudo_labels is not None:
            cluster_labels = torch.as_tensor(
                pseudo_labels, device=self.device, dtype=torch.long
            )
            indicator = indicator & cluster_labels.eq(consensus_prediction)

        indicator = indicator & unlabeled_mask
        bot_idx = self._bot_class_index()
        selected_preds = consensus_prediction[indicator]
        bot_count = int((selected_preds == bot_idx).sum().item()) if indicator.any() else 0
        human_count = int(indicator.sum().item()) - bot_count

        offline_metrics = None
        if indicator.any() and getattr(self, "y", None) is not None:
            y_true = self.y.to(self.device)[indicator]
            y_hat = consensus_prediction[indicator]
            tp = int(((y_hat == bot_idx) & (y_true == bot_idx)).sum().item())
            fp = int(((y_hat == bot_idx) & (y_true != bot_idx)).sum().item())
            fn = int(((y_hat != bot_idx) & (y_true == bot_idx)).sum().item())
            offline_metrics = {
                "pseudo_label_precision": float(tp) / float(tp + fp) if (tp + fp) else 0.0,
                "pseudo_label_recall": float(tp) / float(tp + fn) if (tp + fn) else 0.0,
            }

        stage = int(getattr(self, "_current_stage", 0))
        self._log_stage_ensemble_stats(
            stage=stage,
            member_names=member_labels,
            member_candidate_counts=eval_stats["member_candidate_counts"],
            agreement_count=int(agreement_u.sum().item()),
            confidence_passed_count=int(conf_passed.sum().item()),
            variance_passed_count=int(var_passed.sum().item()),
            final_selected_count=int(indicator.sum().item()),
            bot_count=bot_count,
            human_count=human_count,
            diversity_stats={
                "mean_prediction_variance": eval_stats["mean_prediction_variance"],
                "pseudo_label_coverage": eval_stats["pseudo_label_coverage"],
                "pairwise_disagreement": eval_stats["pairwise_disagreement"],
            },
            offline_metrics=offline_metrics,
        )
        logger.info(
            "[ensemble-filter] type={} mean_p_avg={:.4f} var_avg={:.6f} "
            "agreement_ratio={:.4f} coverage={:.4f} class_dist={{bot:{}, human:{}}}".format(
                self._normalized_ensemble_type(),
                eval_stats["mean_probability_avg"],
                eval_stats["mean_prediction_variance"],
                eval_stats["agreement_ratio"],
                eval_stats["pseudo_label_coverage"],
                bot_count,
                human_count,
            )
        )

        y_train = F.one_hot(self.labels, num_classes=self.num_classes).float()
        y_train[~self.running_train_mask] = 0
        prediction = F.one_hot(
            consensus_prediction, num_classes=self.num_classes
        ).float()
        prediction[self.running_train_mask] = y_train[self.running_train_mask]
        train_mask = deepcopy(self.running_train_mask)
        train_mask[indicator] = True
        y_train[indicator] = prediction[indicator]
        return y_train, train_mask

    def _selftraining_scores(self, confidence_threshold):
        with inference_mode(self.model, dropout=getattr(self.args, "mc_dropout", False)):
            predictions = torch.stack([
                F.softmax(self._logits(self.edge_index, self.edge_type), dim=1)
                for _ in range(20)
            ])
        confidence, predicted = predictions.mean(dim=0).max(dim=1)
        uncertainty = predictions.std(dim=0).gather(1, predicted[:, None]).squeeze(1)
        selected = (confidence >= confidence_threshold) & (uncertainty <= self.args.kappa_p)
        if self._attn_filter_enabled():
            score = self._compute_attention_score()
            selected &= score >= self.args.attention_threshold
        return predicted, selected & self._unlabeled_candidates()

    def _apply_pseudo_labels(self, predicted, selected):
        trained = self.running_train_mask
        targets = F.one_hot(self.labels, num_classes=self.num_classes).float()
        targets[~trained] = 0
        targets[selected] = F.one_hot(predicted[selected], num_classes=self.num_classes).float()
        return targets, trained | selected

    def selftraining_with_checking(self, pseudo_labels, confidece_threshold):
        if self._ensemble_enabled():
            return self._ensemble_selftraining(confidece_threshold, pseudo_labels=pseudo_labels)
        predicted, selected = self._selftraining_scores(confidece_threshold)
        cluster_labels = torch.as_tensor(pseudo_labels, device=self.device, dtype=torch.long)
        return self._apply_pseudo_labels(predicted, selected & cluster_labels.eq(predicted))

    def selftraining(self, confidece_threshold):
        if self._ensemble_enabled():
            return self._ensemble_selftraining(confidece_threshold)
        return self._apply_pseudo_labels(*self._selftraining_scores(confidece_threshold))


class LiProS(nn.Module):
    def __init__(
        self,
        encoder,
        classifier,
        encoder_needs_new_feature=False,
        dual_space=False,
        consist_space=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.encoder_needs_new_feature = encoder_needs_new_feature
        self.dual_space = dual_space
        self.consist_space = consist_space
        self.reset_parameters()

    def represent(self, des, tweet, num, cat, edge_index, edge_type, new_feature=None):
        if self.dual_space or self.consist_space:
            return self.encoder.forward_graph(des, tweet, num, cat, edge_index, edge_type)
        if self.encoder_needs_new_feature:
            if new_feature is None:
                raise ValueError("The current encoder requires new_feature.")
            return self.encoder(des, tweet, num, cat, new_feature, edge_index, edge_type)
        return self.encoder(des, tweet, num, cat, edge_index, edge_type)

    def forward(self, des, tweet, num, cat, edge_index, edge_type, new_feature=None):
        if self.consist_space:
            if new_feature is None:
                raise ValueError("The esa_cons encoder requires new_feature.")
            out = self.encoder.forward_esa(
                des, tweet, num, cat, new_feature, edge_index, edge_type
            )
        elif self.dual_space:
            if new_feature is None:
                raise ValueError("The dual-space encoder requires new_feature.")
            z_graph = self.encoder.forward_graph(des, tweet, num, cat, edge_index, edge_type)
            z_feature = self.encoder.forward_feature(
                des, tweet, num, cat, new_feature, edge_index, edge_type
            )
            out = torch.cat([z_graph, z_feature], dim=1)
        else:
            out = self.represent(des, tweet, num, cat, edge_index, edge_type, new_feature)
        logits, predictions = self.classifier(out)
        return logits, predictions

    def forward_with_reps(self, des, tweet, num, cat, edge_index, edge_type, new_feature=None):
        if not self.consist_space:
            raise RuntimeError("forward_with_reps requires the esa_cons encoder.")
        if new_feature is None:
            raise ValueError("The esa_cons encoder requires new_feature.")
        z_esa = self.encoder.forward_esa(
            des, tweet, num, cat, new_feature, edge_index, edge_type
        )
        z_graph = self.encoder.forward_graph(des, tweet, num, cat, edge_index, edge_type)
        logits, predictions = self.classifier(z_esa)
        return logits, z_esa, z_graph

    def cls(self, des, tweet, num, cat, edge_index, edge_type, new_feature=None):
        return self.forward(des, tweet, num, cat, edge_index, edge_type, new_feature)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.classifier)






def sample_mask(idx, l):
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)
