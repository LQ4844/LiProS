# Fusion code derived from filipovicSamuel's ESA-BotRGCN implementation.
# M3S-derived encoder sources are listed under Acknowledgements in README.md.
from typing import List

import torch
from torch import Tensor
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, HGTConv, RGCNConv, Sequential
import torch.nn as nn
import torch.nn.functional as F

class FourFeatureProjector(nn.Module):

    def __init__(
        self,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        embedding_dimension=128,
    ):
        super().__init__()
        part_dim = embedding_dimension // 4
        self.output_size = part_dim * 4
        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(size, part_dim), nn.LeakyReLU())
                for size in (des_size, tweet_size, num_prop_size, cat_prop_size)
            ]
        )

    def forward(self, des, tweet, num_prop, cat_prop):
        inputs = (des, tweet, num_prop, cat_prop)
        return torch.cat(
            [projection(value) for projection, value in zip(self.projections, inputs)],
            dim=1,
        )

    def reset_parameters(self):
        for projection in self.projections:
            for module in projection:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()


_FourFeatureProjector = FourFeatureProjector


def _resolve_projector(module, projector, des_size, tweet_size, num_prop_size, cat_prop_size, embedding_dimension):
    if projector is None:
        module.projector = FourFeatureProjector(
            des_size, tweet_size, num_prop_size, cat_prop_size, embedding_dimension
        )
        module._external_projector = None
        return module.projector.output_size
    module._external_projector = {"projector": projector}
    return projector.output_size


def _project_features(module, des, tweet, num_prop, cat_prop):
    if getattr(module, "_external_projector", None) is not None:
        return module._external_projector["projector"](des, tweet, num_prop, cat_prop)
    return module.projector(des, tweet, num_prop, cat_prop)


class FeatureGCN(nn.Module):

    def __init__(
        self,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        embedding_dimension=128,
        dropout=0.3,
        projector=None,
    ):
        super().__init__()
        self.input_size = des_size + tweet_size + num_prop_size + cat_prop_size
        self.representation_size = embedding_dimension
        self.dropout = dropout
        in_dim = _resolve_projector(
            self, projector, des_size, tweet_size, num_prop_size, cat_prop_size, embedding_dimension
        )
        self.conv1 = GCNConv(in_dim, embedding_dimension)
        self.conv2 = GCNConv(embedding_dimension, embedding_dimension)
        self.norm = BatchNorm(embedding_dimension)
        self.activation = nn.PReLU()

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        x = _project_features(self, des, tweet, num_prop, cat_prop)
        x = self.activation(self.norm(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class FeatureGAT(nn.Module):

    def __init__(
        self,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        embedding_dimension=128,
        dropout=0.3,
        heads=2,
        projector=None,
    ):
        super().__init__()
        self.input_size = des_size + tweet_size + num_prop_size + cat_prop_size
        self.representation_size = embedding_dimension
        self.dropout = dropout
        in_dim = _resolve_projector(
            self, projector, des_size, tweet_size, num_prop_size, cat_prop_size, embedding_dimension
        )
        self.conv1 = GATConv(
            in_dim,
            embedding_dimension,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            embedding_dimension,
            embedding_dimension,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.norm = BatchNorm(embedding_dimension)
        self.activation = nn.PReLU()

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        x = _project_features(self, des, tweet, num_prop, cat_prop)
        x = self.activation(self.norm(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
    
class RGCN(nn.Module):
    def __init__(self, layer_sizes, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128, batchnorm_mm=0.99):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]

        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        layers = []
        for in_dim, out_dim in zip(layer_sizes[1:-1], layer_sizes[1:]):
            layers.append((RGCNConv(in_dim, out_dim,num_relations=2), 'x, edge_index,edge_type -> x'),)
            layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index,edge_type', layers)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        return self.model(x, edge_index,edge_type)

    def reset_parameters(self):
        self.model.reset_parameters()

class SARGCN(nn.Module):
    def __init__(
        self,
        layer_sizes,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        embedding_dimension=128,
        batchnorm_mm=0.99,
        drop_rate=0.3,
        projector=None,
    ):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self._external_projector = None

        if projector is not None:
            self._external_projector = {"projector": projector}
            self.linear_relu_des = None
            self.linear_relu_tweet = None
            self.linear_relu_num_prop = None
            self.linear_relu_cat_prop = None
        else:
            self.linear_relu_des=nn.Sequential(
                nn.Linear(des_size,int(embedding_dimension/4)),
                nn.LeakyReLU()
            )
            self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.LeakyReLU()
            )
            self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.LeakyReLU()
            )
            self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.LeakyReLU()
            )
        
        layers = []
        for in_dim, out_dim in zip(layer_sizes[1:-1], layer_sizes[1:]):
            layers.append((RGCNConv(in_dim, out_dim,num_relations=2), 'x, edge_index,edge_type -> x'),)
            layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(nn.PReLU())
            layers.append( (nn.Dropout(drop_rate), 'x -> x'), )

        self.model = Sequential('x, edge_index,edge_type', layers)

    def _embed_inputs(self, des, tweet, num_prop, cat_prop):
        if self._external_projector is not None:
            return self._external_projector["projector"](des, tweet, num_prop, cat_prop)
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        return torch.cat((d,t,n,c),dim=1)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index,edge_type):
        x = self._embed_inputs(des, tweet, num_prop, cat_prop)
        return self.model(x, edge_index,edge_type)

    def reset_parameters(self):
        self.model.reset_parameters()
        if self._external_projector is None:
            for module in (
                self.linear_relu_des,
                self.linear_relu_tweet,
                self.linear_relu_num_prop,
                self.linear_relu_cat_prop,
            ):
                reset_linear = getattr(module[0], "reset_parameters", None)
                if callable(reset_linear):
                    reset_linear()


class ESAFeatureAttention(nn.Module):

    def __init__(self, num_features=5):
        super().__init__()
        self.num_features = num_features
        self.attn_weights = nn.Parameter(torch.empty(num_features, 1))
        self.register_buffer("last_alpha", torch.empty(0), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, features: List[Tensor]) -> Tensor:
        stacked = torch.stack(features, dim=0)
        weights = torch.softmax(self.attn_weights, dim=0).unsqueeze(1)
        n = stacked.size(1)
        self.last_alpha = weights.reshape(1, -1).expand(n, -1).detach()
        return (weights * stacked).sum(dim=0)

    def get_attention_weights(self) -> Tensor:
        return torch.softmax(self.attn_weights.detach(), dim=0).reshape(-1)

    def get_node_attention(self) -> Tensor:
        if self.last_alpha.numel() == 0:
            raise RuntimeError("Call ESAFeatureAttention.forward() before reading node attention.")
        return self.last_alpha


class ESAFeatureAttentionPaper(nn.Module):

    def __init__(self, dim, num_features=5):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.W_a = nn.Linear(dim, dim, bias=False)
        self.q = nn.Parameter(torch.empty(dim))
        self.register_buffer("last_alpha_mean", torch.zeros(num_features), persistent=False)
        self.register_buffer("last_alpha", torch.empty(0), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.normal_(self.q, std=0.02)

    def forward(self, features: List[Tensor]) -> Tensor:
        scores = torch.stack(
            [(self.W_a(r_k) * self.q).sum(dim=-1) for r_k in features], dim=1
        )
        alpha = torch.softmax(scores, dim=1)
        stacked = torch.stack(features, dim=1)
        fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        self.last_alpha = alpha.detach()
        self.last_alpha_mean = alpha.mean(dim=0).detach()
        return fused

    def get_attention_weights(self) -> Tensor:
        return self.last_alpha_mean

    def get_node_attention(self) -> Tensor:
        if self.last_alpha.numel() == 0:
            raise RuntimeError("Call ESAFeatureAttentionPaper.forward() before reading node attention.")
        return self.last_alpha


class LiProSEncoder(nn.Module):

    def __init__(
        self,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        new_feature_size=2,
        common_dim=256,
        embedding_dimension=128,
        num_relations=2,
        dropout=0.4,
        paper_fusion=True,
        use_residual_fusion=True,
        backbone="shared",
    ):
        super().__init__()
        self.dropout = dropout
        self.paper_fusion = paper_fusion
        self.use_residual_fusion = use_residual_fusion
        self.backbone = backbone
        if backbone not in ("shared", "residual"):
            raise ValueError("The ESA backbone must be shared or residual.")
        cd = 256 if paper_fusion else common_dim
        self.common_dim = cd
        self.input_size = des_size + tweet_size + num_prop_size + cat_prop_size + new_feature_size
        self.representation_size = embedding_dimension

        def enc(in_dim):
            return nn.Sequential(nn.Linear(in_dim, cd), nn.LeakyReLU())

        self.linear_relu_des = enc(des_size)
        self.linear_relu_tweet = enc(tweet_size)
        self.linear_relu_num_prop = enc(num_prop_size)
        self.linear_relu_cat_prop = enc(cat_prop_size)
        self.linear_relu_new_feature = enc(new_feature_size)

        if paper_fusion:
            self.feature_attention = ESAFeatureAttentionPaper(cd, num_features=5)
        else:
            self.feature_attention = ESAFeatureAttention(num_features=5)

        self.linear_relu_input = nn.Sequential(
            nn.Linear(cd, embedding_dimension),
            nn.LeakyReLU(),
        )
        if backbone == "shared":
            self.rgcn = RGCNConv(
                embedding_dimension, embedding_dimension, num_relations=num_relations
            )
        else:
            self.rgcn1 = RGCNConv(
                embedding_dimension, embedding_dimension, num_relations=num_relations
            )
            self.rgcn2 = RGCNConv(
                embedding_dimension, embedding_dimension, num_relations=num_relations
            )
            self.output_norm = nn.LayerNorm(embedding_dimension)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        feats = [
            self.linear_relu_des(des),
            self.linear_relu_tweet(tweet),
            self.linear_relu_num_prop(num_prop),
            self.linear_relu_cat_prop(cat_prop),
            self.linear_relu_new_feature(new_feature),
        ]
        fused = self.feature_attention(feats)
        if self.use_residual_fusion:
            fused = fused + torch.stack(feats, dim=1).mean(dim=1)
        del feats

        x = self.linear_relu_input(fused)
        if self.backbone == "shared":
            x = self.rgcn(x, edge_index, edge_type)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.rgcn(x, edge_index, edge_type)
        else:
            residual = x
            x = self.rgcn1(x, edge_index, edge_type)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.rgcn2(x, edge_index, edge_type)
            x = self.output_norm(x + residual)
        return self.linear_relu_output1(x)

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def print_attention_weights(self):
        weights = self.feature_attention.get_attention_weights()
        names = ["Description", "Tweets", "NumProp", "CatProp", "NewFeature"]
        print("\n[Attention] Mean weights (alpha) for the five feature streams:")
        for i, name in enumerate(names):
            value = weights[i].item() if weights.numel() > i else float("nan")
            print("  {:<12}: {:.4f}".format(name, value))

    def get_node_attention(self) -> Tensor:
        return self.feature_attention.get_node_attention()


class DualSpaceEncoder(nn.Module):

    def __init__(
        self,
        sargcn_layer_sizes,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        new_feature_size=6,
        common_dim=256,
        embedding_dimension=128,
        num_relations=2,
        dropout=0.3,
        paper_fusion=True,
        use_residual_fusion=True,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.representation_size = embedding_dimension
        self.cls_input_size = embedding_dimension * 2
        self.common_dim = 256 if paper_fusion else common_dim

        self.graph_encoder = SARGCN(
            sargcn_layer_sizes,
            des_size=des_size,
            tweet_size=tweet_size,
            num_prop_size=num_prop_size,
            cat_prop_size=cat_prop_size,
            embedding_dimension=embedding_dimension,
            drop_rate=dropout,
        )
        self.feature_encoder = LiProSEncoder(
            des_size=des_size,
            tweet_size=tweet_size,
            num_prop_size=num_prop_size,
            cat_prop_size=cat_prop_size,
            new_feature_size=new_feature_size,
            common_dim=common_dim,
            embedding_dimension=embedding_dimension,
            num_relations=num_relations,
            dropout=dropout,
            paper_fusion=paper_fusion,
            use_residual_fusion=use_residual_fusion,
        )

    def forward_graph(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        return self.graph_encoder(des, tweet, num_prop, cat_prop, edge_index, edge_type)

    def forward_feature(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        return self.feature_encoder(
            des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type
        )

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        return self.forward_graph(des, tweet, num_prop, cat_prop, edge_index, edge_type)

    def reset_parameters(self):
        if hasattr(self.graph_encoder, "reset_parameters"):
            self.graph_encoder.reset_parameters()
        if hasattr(self.feature_encoder, "reset_parameters"):
            self.feature_encoder.reset_parameters()

    def get_node_attention(self) -> Tensor:
        return self.feature_encoder.get_node_attention()


class ESAConsEncoder(nn.Module):

    def __init__(
        self,
        sargcn_layer_sizes,
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        new_feature_size=6,
        common_dim=256,
        embedding_dimension=128,
        num_relations=2,
        dropout=0.3,
        paper_fusion=True,
        use_residual_fusion=True,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.representation_size = embedding_dimension
        self.common_dim = 256 if paper_fusion else common_dim

        self.esa_encoder = LiProSEncoder(
            des_size=des_size,
            tweet_size=tweet_size,
            num_prop_size=num_prop_size,
            cat_prop_size=cat_prop_size,
            new_feature_size=new_feature_size,
            common_dim=common_dim,
            embedding_dimension=embedding_dimension,
            num_relations=num_relations,
            dropout=dropout,
            paper_fusion=paper_fusion,
            use_residual_fusion=use_residual_fusion,
        )
        self.graph_encoder = SARGCN(
            sargcn_layer_sizes,
            des_size=des_size,
            tweet_size=tweet_size,
            num_prop_size=num_prop_size,
            cat_prop_size=cat_prop_size,
            embedding_dimension=embedding_dimension,
            drop_rate=dropout,
        )

    def forward_esa(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        return self.esa_encoder(
            des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type
        )

    def forward_graph(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        return self.graph_encoder(des, tweet, num_prop, cat_prop, edge_index, edge_type)

    def forward(self, des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type):
        return self.forward_esa(
            des, tweet, num_prop, cat_prop, new_feature, edge_index, edge_type
        )

    def reset_parameters(self):
        if hasattr(self.esa_encoder, "reset_parameters"):
            self.esa_encoder.reset_parameters()
        if hasattr(self.graph_encoder, "reset_parameters"):
            self.graph_encoder.reset_parameters()

    def get_node_attention(self) -> Tensor:
        return self.esa_encoder.get_node_attention()


class HGT(nn.Module):
    def __init__(self, features_num,relation_list,hidden_dimension=256, linear_channels=128, out_channel=32,dropout=0.3):
        super(HGT, self).__init__()

        self.relation_list = list(relation_list)
        self.linear1 = nn.Linear(features_num, hidden_dimension)

        self.HGT_layer1 = HGTConv(in_channels=hidden_dimension, out_channels=hidden_dimension,
                                  metadata=(['user'], self.relation_list))
        self.HGT_layer2 = HGTConv(in_channels=hidden_dimension, out_channels=linear_channels,
                                  metadata=(['user'], self.relation_list))
        self.out1 = torch.nn.Linear(linear_channels, out_channel)
        self.out2 = torch.nn.Linear(out_channel, out_channel)

        self.drop = nn.Dropout(dropout)
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features, edge_index_dict):

        user_features = self.drop(self.ReLU(self.linear1(features)))
        x_dict = {"user": user_features}
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        x_dict = self.HGT_layer2(x_dict, edge_index_dict)
        user_features = self.ReLU(self.out1(x_dict["user"]))
        x = self.out2(user_features)
        return x
    





        
        
        
        
        
            
