import argparse
from ast import literal_eval
from math import isfinite

SUPPORTED_LABEL_RATES = (0.5, 1, 2, 10, 50, 90, 99)
_ENCODER_DEFAULTS = {
    "sargcn": {"lr": 0.01, "decay": 1e-5, "optimizer": "adam"},
    "gcn": {"lr": 0.01, "decay": 1e-5, "optimizer": "adam"},
    "gat": {"lr": 0.01, "decay": 1e-5, "optimizer": "adam"},
    "lipros": {"lr": 1e-3, "decay": 5e-3, "optimizer": "adamw"},
    "dual": {"lr": 1e-3, "decay": 5e-3, "optimizer": "adamw"},
    "esa_cons": {"lr": 1e-3, "decay": 5e-3, "optimizer": "adamw"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LiProS on preprocessed TwiBot-20 or Cresci-15.")
    parser.add_argument("--label-rate", "--label_rate", dest="label_rate", type=float, choices=SUPPORTED_LABEL_RATES, default=50, help="Training label rate in percent. Supported values are 0.5, 1, 2, 10, 50, 90, and 99.")
    parser.add_argument("--folds", type=int, default=1)

    parser.add_argument("--layers", default='[128, 128]', help="Hidden layer widths as a Python-style list, e.g. [128,128]")
    parser.add_argument(
        "--encoder",
        type=str,
        default="lipros",
        choices=["lipros", "sargcn", "gcn", "gat", "dual", "esa_cons"],
        help="lipros uses five-stream attention. sargcn uses the original M3S encoder. gcn and gat use four feature streams. dual concatenates two representations. esa_cons uses consistency regularization.",
    )
    
    parser.add_argument("--epochs", "-e", type=int, default=1000,
                        help="Maximum final training epochs (default: 1000).")
    parser.add_argument("--lr", '-lr', type=float, default=None, help="Learning rate. Defaults to the selected encoder's recommended value.")
    parser.add_argument("--decay", type=float, default=None, help="Weight decay. Defaults to the selected encoder's recommended value.")
    parser.add_argument("--l2-lambda", type=float, default=1e-5,
                        help="Explicit L2 loss coefficient, separate from optimizer weight decay (default: 1e-5).")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["auto", "adam", "adamw"],
        help="auto: lipros/dual/esa_cons->adamw, sargcn->adam",
    )
    parser.add_argument("--patience", type=int, default=200,
                        help="Early-stopping patience in final training (default: 200).")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed for this run")
    parser.add_argument("--deterministic-algorithms", action="store_true",
                        help="Prefer deterministic PyTorch operations and warn about unsupported GPU operations.")
    parser.add_argument("--num-threads", type=int,
                        help="Intra-op CPU threads. Defaults to PyTorch's setting.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed for data splits. Set it independently of the model seed to compare runs on the same splits.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Optional JSON output for run metrics used by hyperparameter search scripts",
    )
    split_options = parser.add_mutually_exclusive_group()
    split_options.add_argument(
        "--split-mode", choices=("auto", "random", "official"), default="auto",
        help="auto uses official splits for Cresci-15 with split metadata and random splits otherwise. "
             "random assigns 10%% of the remaining labeled nodes to validation and the rest to test.",
    )
    split_options.add_argument(
        "--use-original-splits", dest="split_mode", action="store_const", const="official",
        help="Alias for --split-mode official. Keep the source dev/test splits.",
    )
    parser.add_argument(
        "--pseudo-pool", choices=("vu", "all"), default="vu",
        help="vu excludes the current fold's validation and test nodes from pseudo-labeling. "
             "all allows them as candidates.",
    )
    parser.add_argument(
        "--holdout-support", type=int, default=0, metavar="COUNT",
        help="Hide the labels of COUNT accounts as support before creating random splits.",
    )
    parser.add_argument("--epoch-csv", help="Optional epoch CSV for pretraining and final training in every fold and trial.")
    parser.add_argument("--checkpoint-dir", help="Optional directory for the selected model weights from each fold and trial.")
        
    parser.add_argument("--stage", type=int, default=200,
                        help="Maximum number of self-training stages (default: 200).")
    parser.add_argument("--min-unlabeled", type=int, default=1,
                        help="Minimum number of unlabeled candidates to start a self-training stage (default: 1). "
                             "Clustering also requires at least --num_K candidates.")
    parser.add_argument("--clustering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_K", type=int, default=200)
    parser.add_argument("--device", '-d', type=int, default=0, help="CUDA device index, or -1 for CPU")    
    parser.add_argument("--kappa-p", default=0.3, type=float,
                        help="Maximum predicted-class probability standard deviation for a pseudo-label (default: %(default)s).")
    parser.add_argument(
        "--mc-dropout",
        action="store_true",
        help="Enable Monte Carlo dropout for legacy runs. For newer ensemble runs, use --uncertainty-method ensemble.",
    )
    parser.add_argument(
        "--uncertainty-method",
        choices=["legacy", "ensemble"],
        default="legacy",
        help="legacy uses the original 20-pass uncertainty-based pseudo-label selection. ensemble uses one forward pass per member and requires unanimous agreement.",
    )
    parser.add_argument(
        "--ensemble-type",
        choices=[
            "architecture",
            "seed",
            "gnn_hetero",
            "same_init",
            "existing_hetero",
        ],
        default="architecture",
        help="architecture/gnn_hetero uses GCN, GAT, and SARGCN. "
             "seed uses three SARGCN models with different random seeds. "
             "same_init uses the encoder selected by --encoder with different seeds. existing_hetero uses ESA, SARGCN, and Dual.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Number of same_init ensemble members. architecture and seed each use three members.",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Independent random seeds for the three SARGCN members in --ensemble-type seed",
    )
    parser.add_argument(
        "--ensemble-seed-offsets",
        type=int,
        nargs="+",
        default=[0, 1000, 2000],
        help="Seed offsets for same_init members, relative to the base seed for the fold and stage",
    )
    parser.add_argument(
        "--ensemble-reinit-each-stage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reinitialize model weights at each self-training stage (default: continue training).",
    )
    parser.add_argument(
        "--ensemble-variance-threshold",
        type=float,
        default=0.01,
        help="Maximum ensemble variance in the predicted bot probability for accepting a pseudo-label",
    )
    parser.add_argument(
        "--bot-class-index",
        type=int,
        choices=(0, 1),
        default=1,
        help="Bot class index in the logits and softmax output (1 for bots and 0 for humans by default).",
    )

    parser.add_argument("--relation_select", type=int, choices=(0, 1), default=[0, 1], nargs="+",
                        help="Relation IDs for the heterogeneous graph view (0 and 1, mapped by dataset).")
    parser.add_argument("--pretrain_epochs", type=int, default=200,
                        help="Training epochs per self-training stage (default: 200).")

    parser.add_argument("--fusion-dim", type=int, default=256, help="Alignment dimension D for the five feature streams. Paper-style fusion requires D = 256.")
    parser.add_argument("--dropout", "--esa-dropout", dest="esa_dropout", type=float, default=0.4,
                        help="Dropout probability during training and Monte Carlo dropout (default: 0.4).")
    parser.add_argument(
        "--backbone", "--esa-backbone", dest="esa_backbone",
        choices=["shared", "residual"],
        default="shared",
        help="shared applies the same RGCN layer twice. residual uses two separate RGCN layers with a residual connection and layer normalization.",
    )
    parser.add_argument(
        "--no-paper-fusion",
        action="store_true",
        help="Use simplified attention with shared scalar weights. The default uses per-node attention with q and W_a.",
    )
    parser.add_argument(
        "--no-fusion-residual",
        action="store_true",
        help="Disable the residual connection in feature fusion. By default, the mean of the five aligned feature streams is added to the attention output.",
    )
    parser.add_argument(
        "--consist-lambda",
        type=float,
        default=0.1,
        help="Consistency loss weight for esa_cons in L = CE + λ * MSE(norm(z_esa), norm(z_graph))",
    )

    parser.add_argument("--data-file", required=True,
                        help="Feature file produced by python -m src.feature_preprocessing")

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "bo"],
        help="single runs multi-stage training with a fixed confidence threshold. bo tunes the threshold using Gaussian-process Bayesian optimization.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.65, help="Pseudo-label confidence threshold in single mode")
    parser.add_argument(
        "--reinit-final",
        action="store_true",
        help="Reinitialize the model after pseudo-labeling and train on the final labeled set.",
    )
    parser.add_argument(
        "--use-attention-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also filter pseudo-labels by attention score. Requires an encoder that exposes attention weights.",
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=0.3,
        help="Attention score threshold γ, where score = 1 - H_norm(α). Used only with --use-attention-filter.",
    )
    parser.add_argument(
        "--calibrate-decision-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calibrate the bot classification threshold each epoch by maximizing the sum of F1 and accuracy on the validation set.",
    )
    parser.add_argument("--decision-threshold-lower", type=float, default=0.30)
    parser.add_argument("--decision-threshold-upper", type=float, default=0.90)
    parser.add_argument("--decision-threshold-steps", type=int, default=121)
    parser.add_argument("--bo-lower", type=float, default=0.5, help="Lower confidence threshold bound for Bayesian optimization")
    parser.add_argument("--bo-upper", type=float, default=0.9, help="Upper confidence threshold bound for Bayesian optimization")
    parser.add_argument("--bo-initial-points", type=int, default=3, help="Number of initial evaluations before fitting the Gaussian process")
    parser.add_argument("--bo-iterations", type=int, default=2,
                        help="Number of adaptive Bayesian optimization iterations (default: %(default)s).")
    parser.add_argument("--bo-candidates", type=int, default=512, help="Number of random candidates scored by the acquisition function per iteration")
    parser.add_argument("--bo-xi", type=float, default=0.01, help="Exploration parameter for the expected improvement acquisition function")
    parser.add_argument(
        "--bo-metric",
        choices=["val_acc", "val_f1"],
        default="val_f1",
        help="Validation metric used to optimize the confidence threshold",
    )
    parser.add_argument("--bo-seed", type=int, default=0, help="Random seed for Bayesian optimization")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Low-memory smoke run: fewer stages/epochs/clusters/BO iterations",
    )

    args = parser.parse_args()
    args.use_original_splits = args.split_mode == "official"

    try:
        args.layers = literal_eval(args.layers)
    except (SyntaxError, ValueError):
        parser.error("--layers must be a list of positive integers")
    if (not isinstance(args.layers, list) or not args.layers
            or any(type(width) is not int or width <= 0 for width in args.layers)):
        parser.error("--layers must be a list of positive integers")
    if not 1 <= args.folds <= 20 or min(args.epochs, args.stage, args.pretrain_epochs, args.patience, args.num_K) < 1:
        parser.error("--folds must be in [1, 20] and training sizes must be positive")
    if args.min_unlabeled < 1:
        parser.error("--min-unlabeled must be positive")
    if args.holdout_support < 0:
        parser.error("--holdout-support must be non-negative")
    if not isfinite(args.l2_lambda) or args.l2_lambda < 0:
        parser.error("--l2-lambda must be non-negative and finite")
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be positive")
    _ENSEMBLE_ALIASES = {
        "architecture": "architecture",
        "gnn_hetero": "architecture",
        "seed": "seed",
        "same_init": "same_init",
        "existing_hetero": "existing_hetero",
    }
    args.ensemble_type = _ENSEMBLE_ALIASES.get(args.ensemble_type, args.ensemble_type)

    defaults = _ENCODER_DEFAULTS[args.encoder]
    if args.uncertainty_method == "ensemble" and args.ensemble_type in (
        "architecture",
        "seed",
    ):
        defaults = _ENCODER_DEFAULTS["sargcn"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.decay is None:
        args.decay = defaults["decay"]
    if args.optimizer == "auto":
        args.optimizer = defaults["optimizer"]
    args.paper_fusion = not args.no_paper_fusion
    args.fusion_residual = not args.no_fusion_residual

    if not 0.0 < args.decision_threshold_lower < args.decision_threshold_upper < 1.0:
        parser.error("Decision threshold bounds must satisfy 0 < lower < upper < 1.")
    if args.decision_threshold_steps < 2:
        parser.error("--decision-threshold-steps must be at least 2.")

    if args.uncertainty_method == "ensemble":
        if args.ensemble_type in ("architecture", "seed", "existing_hetero"):
            expected_size = 3
        else:
            expected_size = args.ensemble_size
        if expected_size < 2:
            parser.error("--ensemble-size must be at least 2.")
        if args.ensemble_type == "seed":
            if len(args.ensemble_seeds) < expected_size:
                parser.error("--ensemble-seeds must provide at least one seed per ensemble member.")
            args.ensemble_seeds = list(args.ensemble_seeds[:expected_size])
            if len(set(args.ensemble_seeds)) < expected_size:
                parser.error("--ensemble-seeds must be distinct to give members different initializations.")
        elif args.ensemble_type == "same_init":
            if len(args.ensemble_seed_offsets) < expected_size:
                parser.error("--ensemble-seed-offsets must provide at least one offset per ensemble member.")
        if args.mc_dropout:
            parser.error("Remove --mc-dropout when using --uncertainty-method ensemble.")

    if args.debug:
        args.stage = 1
        args.epochs = 50
        args.patience = 20
        args.num_K = 20
        args.pretrain_epochs = 20
        args.bo_initial_points = 2
        args.bo_iterations = 1
        args.bo_candidates = 64
        print(
            "[debug] stage=1 epochs=50 patience=20 num_K=20 "
            "pretrain_epochs=20 bo_initial_points=2 bo_iterations=1 bo_candidates=64"
        )
    print(
        "[config] encoder={} optimizer={} lr={} decay={} mode={} "
        "uncertainty={} ensemble={} reinit_each_stage={} attn_filter={} γ={}".format(
            args.encoder, args.optimizer, args.lr, args.decay, args.mode,
            args.uncertainty_method, args.ensemble_type,
            args.ensemble_reinit_each_stage,
            args.use_attention_filter, args.attention_threshold,
        )
    )
    return args
