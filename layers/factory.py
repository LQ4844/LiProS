from .Classifier import Classifier, DualClassifier
from .GNN import DualSpaceEncoder, ESAConsEncoder, FeatureGAT, FeatureGCN, LiProSEncoder, SARGCN


def build_encoder(args, name, feature_widths, num_relations, num_classes, projector=None):
    des, tweet, numeric, categorical, auxiliary = feature_widths
    hidden = args.layers[-1]
    profile_inputs = {
        "des_size": des,
        "tweet_size": tweet,
        "num_prop_size": numeric,
        "cat_prop_size": categorical,
        "embedding_dimension": hidden,
    }
    lifecycle_inputs = {
        **profile_inputs,
        "new_feature_size": auxiliary,
        "common_dim": args.fusion_dim,
        "num_relations": num_relations,
        "dropout": args.esa_dropout,
        "paper_fusion": args.paper_fusion,
        "use_residual_fusion": args.fusion_residual,
    }
    layer_sizes = [des + tweet + numeric + categorical, *args.layers]

    if name == "lipros":
        encoder = LiProSEncoder(**lifecycle_inputs, backbone=args.esa_backbone)
    elif name == "dual":
        encoder = DualSpaceEncoder(**lifecycle_inputs, sargcn_layer_sizes=layer_sizes)
        classifier = DualClassifier(encoder.cls_input_size, num_classes, hidden, args.esa_dropout)
        return encoder, classifier
    elif name == "esa_cons":
        encoder = ESAConsEncoder(**lifecycle_inputs, sargcn_layer_sizes=layer_sizes)
    elif name == "gcn":
        encoder = FeatureGCN(**profile_inputs, dropout=args.esa_dropout, projector=projector)
    elif name == "gat":
        encoder = FeatureGAT(**profile_inputs, dropout=args.esa_dropout, heads=2, projector=projector)
    elif name == "sargcn":
        encoder = SARGCN(layer_sizes, **profile_inputs, drop_rate=args.esa_dropout, projector=projector)
    else:
        raise ValueError(f"Unknown encoder: {name}")
    return encoder, Classifier(hidden, num_classes)
