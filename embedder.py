from contextlib import contextmanager
import logging

import torch

from layers.factory import build_encoder
from src.data import prepare_training_splits
from src.dataset import CRESCI_RELATION_NAMES, RELATION_NAMES, UNLABELED, load_dataset
from src.feature_preprocessing import combine_auxiliary_features
from src.utils import config2string


logger = logging.getLogger(__name__)


@contextmanager
def inference_mode(model, dropout=False):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.train(dropout)
        if dropout:
            for module, _ in modes:
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


class Embedder:
    def __init__(self, args):
        self.args = args
        if getattr(args, "bot_class_index", 1) not in (0, 1):
            raise ValueError("The bot class index must be 0 or 1 for binary predictions.")

        self.device = (
            torch.device(f"cuda:{args.device}")
            if args.device >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        elif args.device >= 0:
            logger.info("[device] CUDA unavailable, using CPU")

        self.config_str = config2string(args)
        logger.info(f"[LiProS] device={self.device} | {self.config_str}")

        self._load_data()
        self.hidden_layers = args.layers
        self.rep_size = self.hidden_layers[-1]
        self.num_classes = 2
        self.num_relations = len(self.relation_names)
        self.dual_space = (args.encoder == "dual")
        self.consist_space = (args.encoder == "esa_cons")
        self.encoder_needs_new_feature = args.encoder in ("lipros", "dual", "esa_cons")
        self.encoder, self.classifier = self._build_encoder()
        if self.consist_space:
            logger.info(
                "[encoder] ESACons | ESA→z_esa (classification) + SARGCN→z_graph (clustering) | "
                "L=CE+λ·MSE | λ={} | des={} tweet={} num={} cat={} new={} | "
                "D={} | z_dim={} | paper-style fusion={} | fusion residual={}".format(
                    getattr(args, "consist_lambda", 0.1),
                    self.des.size(1), self.tweet.size(1), self.num.size(1),
                    self.cat.size(1), self.new_feature.size(1),
                    self.encoder.common_dim, self.rep_size,
                    args.paper_fusion, args.fusion_residual,
                )
            )
        elif self.dual_space:
            logger.info(
                "[encoder] DualSpace | SARGCN→z_graph (clustering) + ESA→z_feature (classification) | "
                "cls=concat→MLP | des={} tweet={} num={} cat={} new={} | "
                "D={} | z_dim={} | cls_in={} | paper-style fusion={} | fusion residual={}".format(
                    self.des.size(1), self.tweet.size(1), self.num.size(1),
                    self.cat.size(1), self.new_feature.size(1),
                    self.encoder.common_dim, self.rep_size,
                    getattr(self.encoder, "cls_input_size", self.rep_size * 2),
                    args.paper_fusion, args.fusion_residual,
                )
            )
        elif self.encoder_needs_new_feature:
            logger.info(
                "[encoder] LiProS | five feature stream dimensions "
                "des={} tweet={} num={} cat={} new={} "
                "| aligned dimension D={} | representation dimension={} | paper-style fusion={} | fusion residual={}".format(
                    self.des.size(1), self.tweet.size(1), self.num.size(1),
                    self.cat.size(1), self.new_feature.size(1),
                    self.encoder.common_dim, self.rep_size,
                    args.paper_fusion, args.fusion_residual,
                )
            )
        elif args.encoder == "sargcn":
            logger.info("[encoder] SARGCN (original M3S four-stream concatenation)")
        else:
            logger.info("[encoder] {} (graph model with four feature streams)".format(args.encoder.upper()))

    def _load_data(self):
        args = self.args
        dataset = load_dataset(args.data_file)
        self.relation_names = (
            CRESCI_RELATION_NAMES if dataset.get("dataset") == "cresci-15" else RELATION_NAMES
        )
        labels = dataset["labels"]
        self.split_masks, scope = prepare_training_splits(dataset, args)
        self.pseudo_label_scope = scope.to(self.device)
        self.num_nodes = labels.numel()
        if args.clustering and args.num_K > self.num_nodes:
            raise ValueError(f"Number of clusters ({args.num_K}) exceeds {self.num_nodes} accounts.")
        self.des, self.tweet, self.num, self.cat = (
            dataset[name].to(self.device) for name in ("description", "tweets", "numeric", "categorical")
        )
        self.input_size = sum(stream.size(1) for stream in (self.des, self.tweet, self.num, self.cat))
        self.y = labels.to(self.device)
        self.edge_index = dataset["edge_index"].to(self.device)
        self.edge_type = dataset["edge_type"].to(self.device)
        self.new_feature = combine_auxiliary_features(dataset["lifecycle"]).to(self.device).float()
        nodes = torch.arange(self.num_nodes, device=self.device)
        self.cnn_edge_index = torch.stack((nodes, nodes))
        self.cnn_edge_type = torch.ones_like(nodes)

        self.dataset_metadata = {
            "name": dataset.get("dataset"),
            "format_version": dataset["format_version"],
            "source_split_sizes": dataset.get("split_sizes"),
            "nodes": self.num_nodes,
            "unlabeled_nodes": int(labels.eq(UNLABELED).sum()),
            "directed_edges": self.edge_type.numel(),
        }
        logger.info("[data] %s accounts (%s unlabeled), %s directed edges",
                    self.num_nodes, self.dataset_metadata["unlabeled_nodes"], self.edge_type.numel())

    @property
    def new_edge_index(self):
        return {
            ("user", self.relation_names[index], "user"): self.edge_index[:, self.edge_type == index]
            for index in self.args.relation_select
        }

    def _build_encoder(self, encoder_name=None, shared_projector=None):
        widths = tuple(stream.size(1) for stream in (
            self.des, self.tweet, self.num, self.cat, self.new_feature,
        ))
        return build_encoder(
            self.args, encoder_name or self.args.encoder, widths,
            self.num_relations, self.num_classes, shared_projector,
        )

    def _evaluation_predictions(self, des, tweet, num, cat, edge_index, edge_type):
        with inference_mode(self.model):
            return self.model.cls(
                des, tweet, num, cat, edge_index, edge_type,
                new_feature=self.new_feature,
            )[1]

    def _evaluation_bot_probabilities(self, des, tweet, num, cat, edge_index, edge_type):
        with inference_mode(self.model):
            logits = self.model.cls(
                des, tweet, num, cat, edge_index, edge_type,
                new_feature=self.new_feature,
            )[0]
        bot_index = int(getattr(self.args, "bot_class_index", 1))
        return torch.softmax(logits, dim=1)[:, bot_index]

# Preserve imports used by earlier releases.
embedder = Embedder
