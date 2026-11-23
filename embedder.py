import copy
import torch
from src.utils import config2string
from src.utils import compute_accuracy, select_binary_threshold
from src.data import create_masks
from src.feature_preprocessing import combine_auxiliary_features
from layers import (
    Classifier, DualClassifier, FeatureGAT, FeatureGCN, FourFeatureProjector,
    RGCN, SARGCN, HGT, LiProSEncoder, DualSpaceEncoder, ESAConsEncoder,
)
from torch_geometric.utils import to_scipy_sparse_matrix,from_scipy_sparse_matrix,to_edge_index,unbatch_edge_index
import numpy as np
import scipy.sparse as sp 
from src.dataset import load_dataset

class embedder:
    def __init__(self, args):
        self.args = args

        self.device = (
            torch.device(f"cuda:{args.device}")
            if args.device >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        elif args.device >= 0:
            print("[device] CUDA unavailable, using CPU")

        self.config_str = config2string(args)
        print(f"[LiProS] device={self.device} | {self.config_str}")

        dataset = load_dataset(args.data_file)
        des_tensor, tweets_tensor, num_prop, category_prop, labels, edge_index, edge_type = (
            dataset[name].to(self.device)
            for name in ("description", "tweets", "numeric", "categorical",
                         "labels", "edge_index", "edge_type")
        )
        x = torch.cat((des_tensor, tweets_tensor, num_prop, category_prop), dim=1)
        print(f"[data] {x.size(0)} accounts, {edge_index.size(1)} directed edges")
        numpy_random_state = np.random.get_state()
        try:
            np.random.seed(int(getattr(args, "split_seed", 0)))
            self.split_masks = create_masks(x, labels, args.label_rate)
        finally:
            np.random.set_state(numpy_random_state)
        self.x=x
        self.des=des_tensor
        self.tweet=tweets_tensor
        self.num=num_prop
        self.cat=category_prop 
        self.y=labels
        self.edge_index=edge_index
        self.edge_type=edge_type
        self.num_nodes=x.shape[0]
        if args.clustering and args.num_K > self.num_nodes:
            raise ValueError(f'Number of clusters ({args.num_K}) exceeds {self.num_nodes} accounts.')
        self.new_feature = combine_auxiliary_features(dataset["lifecycle"]).to(self.device).float()
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=self.num_nodes)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj_cnn = self.normalize(sp.eye(adj.shape[0]))
        self.cnn_edge_index=from_scipy_sparse_matrix(adj_cnn)[0]
        self.cnn_edge_index=self.cnn_edge_index.to(self.device)
        self.cnn_edge_type=from_scipy_sparse_matrix(adj_cnn)[1]
        self.cnn_edge_type=self.cnn_edge_type.long().to(self.device)

        self.hidden_layers = args.layers
        input_size = self.x.size(1)
        rep_size = self.hidden_layers[-1]
      
        self.unique_labels = self.y.unique()
        self.num_classes = len(self.unique_labels)

        self.new_edge_index={}
        relation_dict = {
        0:'followers',
        1:'followings',
        }
        if len(args.relation_select) > 0:
            for index in args.relation_select:
                self.new_edge_index[("user", relation_dict[index], "user")] = self.edge_index[:,self.edge_type==index]

        self.num_relations = max(int(self.edge_type.max().item()) + 1, 2) if self.edge_type.numel() else 2
        self.rep_size = rep_size
        self.input_size = input_size
        self.dual_space = (args.encoder == "dual")
        self.consist_space = (args.encoder == "esa_cons")
        self.encoder_needs_new_feature = args.encoder in ("lipros", "dual", "esa_cons")
        self.encoder, self.classifier = self._build_encoder()
        if self.consist_space:
            print(
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
            print(
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
            print(
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
            print("[encoder] SARGCN (original M3S four-stream concatenation)")
        else:
            print("[encoder] {} (graph model with four feature streams)".format(args.encoder.upper()))

        self.cnt = 0
        self.cnt_precision = 0
        self.cnt_recall = 0
        self.cnt_f1 = 0

        self.best_val = 0
        self.best_val_precision = 0
        self.best_val_recall = 0
        self.best_val_f1 = 0

        self.epoch_acc_list = []
        self.epoch_precision_list = []
        self.epoch_recall_list = []
        self.epoch_f1_list = []

        self.train_accs = [] ; self.valid_accs = [] ; self.test_accs = []
        self.train_precisions = [] ; self.valid_precisions = [] ; self.test_precisions = []
        self.train_recalls = [] ; self.valid_recalls = [] ; self.test_recalls = []
        self.train_f1s = [] ; self.valid_f1s = [] ; self.test_f1s = []

        self.running_train_accs = [] ; self.running_valid_accs = [] ; self.running_test_accs = []
        self.running_train_precisions = [] ; self.running_valid_precisions = [] ; self.running_test_precisions = []
        self.running_train_recalls = [] ; self.running_valid_recalls = [] ; self.running_test_recalls = []
        self.running_train_f1s = [] ; self.running_valid_f1s = [] ; self.running_test_f1s = []
        self.running_decision_thresholds = []
        self.selected_decision_threshold = 0.5

    def _build_encoder(self, encoder_name=None, shared_projector=None):
        encoder_name = encoder_name or self.args.encoder
        if encoder_name == "lipros":
            encoder = LiProSEncoder(
                des_size=self.des.size(1),
                tweet_size=self.tweet.size(1),
                num_prop_size=self.num.size(1),
                cat_prop_size=self.cat.size(1),
                new_feature_size=self.new_feature.size(1),
                common_dim=self.args.fusion_dim,
                embedding_dimension=self.rep_size,
                num_relations=self.num_relations,
                dropout=self.args.esa_dropout,
                paper_fusion=self.args.paper_fusion,
                use_residual_fusion=self.args.fusion_residual,
                backbone=self.args.esa_backbone,
            )
            classifier = Classifier(self.rep_size, self.num_classes)
        elif encoder_name == "dual":
            encoder = DualSpaceEncoder(
                sargcn_layer_sizes=[self.input_size] + self.hidden_layers,
                des_size=self.des.size(1),
                tweet_size=self.tweet.size(1),
                num_prop_size=self.num.size(1),
                cat_prop_size=self.cat.size(1),
                new_feature_size=self.new_feature.size(1),
                common_dim=self.args.fusion_dim,
                embedding_dimension=self.rep_size,
                num_relations=self.num_relations,
                dropout=self.args.esa_dropout,
                paper_fusion=self.args.paper_fusion,
                use_residual_fusion=self.args.fusion_residual,
            )
            classifier = DualClassifier(
                encoder.cls_input_size,
                self.num_classes,
                mid_size=self.rep_size,
                dropout=self.args.esa_dropout,
            )
        elif encoder_name == "esa_cons":
            encoder = ESAConsEncoder(
                sargcn_layer_sizes=[self.input_size] + self.hidden_layers,
                des_size=self.des.size(1),
                tweet_size=self.tweet.size(1),
                num_prop_size=self.num.size(1),
                cat_prop_size=self.cat.size(1),
                new_feature_size=self.new_feature.size(1),
                common_dim=self.args.fusion_dim,
                embedding_dimension=self.rep_size,
                num_relations=self.num_relations,
                dropout=self.args.esa_dropout,
                paper_fusion=self.args.paper_fusion,
                use_residual_fusion=self.args.fusion_residual,
            )
            classifier = Classifier(self.rep_size, self.num_classes)
        elif encoder_name == "gcn":
            encoder = FeatureGCN(
                des_size=self.des.size(1),
                tweet_size=self.tweet.size(1),
                num_prop_size=self.num.size(1),
                cat_prop_size=self.cat.size(1),
                embedding_dimension=self.rep_size,
                dropout=self.args.esa_dropout,
                projector=shared_projector,
            )
            classifier = Classifier(self.rep_size, self.num_classes)
        elif encoder_name == "gat":
            encoder = FeatureGAT(
                des_size=self.des.size(1),
                tweet_size=self.tweet.size(1),
                num_prop_size=self.num.size(1),
                cat_prop_size=self.cat.size(1),
                embedding_dimension=self.rep_size,
                dropout=self.args.esa_dropout,
                heads=2,
                projector=shared_projector,
            )
            classifier = Classifier(self.rep_size, self.num_classes)
        elif encoder_name == "sargcn":
            encoder = SARGCN(
                [self.input_size] + self.hidden_layers,
                des_size=self.des.size(1),
                tweet_size=self.tweet.size(1),
                num_prop_size=self.num.size(1),
                cat_prop_size=self.cat.size(1),
                embedding_dimension=self.rep_size,
                drop_rate=self.args.esa_dropout,
                projector=shared_projector,
            )
            classifier = Classifier(self.rep_size, self.num_classes)
        else:
            raise ValueError("Unknown encoder: {}".format(encoder_name))
        return encoder, classifier

    def reset_run_state(self):
        self.best_val = 0
        self.best_val_precision = 0
        self.best_val_recall = 0
        self.best_val_f1 = 0
        self.cnt = 0
        self.cnt_precision = 0
        self.cnt_recall = 0
        self.cnt_f1 = 0

        self.epoch_acc_list = []
        self.epoch_precision_list = []
        self.epoch_recall_list = []
        self.epoch_f1_list = []

        self.train_accs = [] ; self.valid_accs = [] ; self.test_accs = []
        self.train_precisions = [] ; self.valid_precisions = [] ; self.test_precisions = []
        self.train_recalls = [] ; self.valid_recalls = [] ; self.test_recalls = []
        self.train_f1s = [] ; self.valid_f1s = [] ; self.test_f1s = []

        self.running_train_accs = [] ; self.running_valid_accs = [] ; self.running_test_accs = []
        self.running_train_precisions = [] ; self.running_valid_precisions = [] ; self.running_test_precisions = []
        self.running_train_recalls = [] ; self.running_valid_recalls = [] ; self.running_test_recalls = []
        self.running_train_f1s = [] ; self.running_valid_f1s = [] ; self.running_test_f1s = []
        self.running_decision_thresholds = []
        self.selected_decision_threshold = 0.5

    def normalize(self,mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

        
    def _evaluation_predictions(self, des, tweet, num, cat, edge_index, edge_type):
        self.model.eval()
        return self.model.cls(
            des, tweet, num, cat, edge_index, edge_type,
            new_feature=self.new_feature,
        )[1]

    def _evaluation_bot_probabilities(self, des, tweet, num, cat, edge_index, edge_type):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.cls(
                des, tweet, num, cat, edge_index, edge_type,
                new_feature=self.new_feature,
            )[0]
        bot_index = int(getattr(self.args, "bot_class_index", 1))
        return torch.softmax(logits, dim=1)[:, bot_index]

    def evaluate(self, des, tweet, num, cat, y, edge_index,edge_type, st):

        decision_threshold = 0.5
        if getattr(self.args, "calibrate_decision_threshold", False):
            bot_probability = self._evaluation_bot_probabilities(
                des, tweet, num, cat, edge_index, edge_type
            )
            bot_index = int(getattr(self.args, "bot_class_index", 1))
            decision_threshold, _, _ = select_binary_threshold(
                bot_probability[self.val_mask],
                y[self.val_mask],
                lower=float(getattr(self.args, "decision_threshold_lower", 0.3)),
                upper=float(getattr(self.args, "decision_threshold_upper", 0.9)),
                steps=int(getattr(self.args, "decision_threshold_steps", 121)),
                positive_index=bot_index,
            )
            preds = torch.full_like(y, fill_value=1 - bot_index)
            preds[bot_probability >= decision_threshold] = bot_index
            st += '| decision_threshold: {:.3f} '.format(decision_threshold)
        else:
            preds = self._evaluation_predictions(des, tweet, num, cat, edge_index, edge_type)
        self.running_decision_thresholds.append(float(decision_threshold))
        train_acc, val_acc, test_acc, train_precision, val_precision,test_precision, train_recall, val_recall, test_recall, train_f1,val_f1, test_f1= compute_accuracy(preds, y, self.train_mask, self.val_mask, self.test_mask)
        self.running_train_accs.append(train_acc) ; self.running_valid_accs.append(val_acc) ; self.running_test_accs.append(test_acc)        
        self.running_train_precisions.append(train_precision) ; self.running_valid_precisions.append(val_precision) ; self.running_test_precisions.append(test_precision)   
        self.running_train_recalls.append(train_recall) ; self.running_valid_recalls.append(val_recall) ; self.running_test_recalls.append(test_recall)   
        self.running_train_f1s.append(train_f1) ; self.running_valid_f1s.append(val_f1) ; self.running_test_f1s.append(test_f1)   

        if val_acc > self.best_val:
            self.best_val = val_acc
            self.cnt = 0
        else:
            self.cnt += 1

        if val_precision > self.best_val_precision:
            self.best_val_precision = val_precision
            self.cnt_precision = 0
        else:
            self.cnt_precision += 1

        if val_recall > self.best_val_recall:
            self.best_val_recall = val_recall
            self.cnt_recall = 0
        else:
            self.cnt_recall += 1

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.cnt_f1 = 0
        else:
            self.cnt_f1 += 1
        
        st += '| train_acc: {:.2f} | valid_acc : {:.2f} | test_acc : {:.2f} | train_precision: {:.2f} | val_precision: {:.2f}| test_precision: {:.2f}| train_recall: {:.2f}| val_recall: {:.2f}| test_recall: {:.2f} | train_f1: {:.2f} | val_f1 : {:.2f} | test_f1 : {:.2f} '.format(train_acc, val_acc, test_acc, train_precision, val_precision,test_precision, train_recall, val_recall, test_recall, train_f1,val_f1, test_f1)
        print(st)
        return train_acc,val_acc,test_acc,train_precision,val_precision,test_precision,train_recall,val_recall,test_recall,train_f1,val_f1,test_f1
        
    def save_results(self, fold):

        train_acc, val_acc, test_acc = torch.tensor(self.running_train_accs), torch.tensor(self.running_valid_accs), torch.tensor(self.running_test_accs)        
        train_precision, val_precision, test_precision = torch.tensor(self.running_train_precisions), torch.tensor(self.running_valid_precisions), torch.tensor(self.running_test_precisions)    
        train_recall, val_recall, test_recall = torch.tensor(self.running_train_recalls), torch.tensor(self.running_valid_recalls), torch.tensor(self.running_test_recalls)        
        train_f1, val_f1, test_f1 = torch.tensor(self.running_train_f1s), torch.tensor(self.running_valid_f1s), torch.tensor(self.running_test_f1s)     
               
        select_score = val_f1 + val_acc
        selected_epoch = int(select_score.argmax().item())
        self.selected_decision_threshold = self.running_decision_thresholds[selected_epoch]

        best_train_acc = train_acc[selected_epoch]
        best_val_acc = val_acc[selected_epoch]
        best_test_acc = test_acc[selected_epoch]

        best_train_precision = train_precision[selected_epoch]
        best_val_precision = val_precision[selected_epoch]
        best_test_precison = test_precision[selected_epoch]

        best_train_recall = train_recall[selected_epoch]
        best_val_recall = val_recall[selected_epoch]
        best_test_recall = test_recall[selected_epoch]

        best_train_f1 = train_f1[selected_epoch]
        best_val_f1 = val_f1[selected_epoch]
        best_test_f1 = test_f1[selected_epoch]

        print(
            "[Fold : {}] select epoch {}/{} by val_f1+val_acc={:.2f} "
            "(val_acc={:.2f}, val_f1={:.2f}, val_p={:.2f}, val_r={:.2f}) "
            "| test_acc={:.2f} test_p={:.2f} test_r={:.2f} test_f1={:.2f}".format(
                fold + 1,
                selected_epoch + 1,
                len(select_score),
                float(select_score[selected_epoch]),
                float(best_val_acc),
                float(best_val_f1),
                float(best_val_precision),
                float(best_val_recall),
                float(best_test_acc),
                float(best_test_precison),
                float(best_test_recall),
                float(best_test_f1),
            )
        )

        self.epoch_acc_list.append(selected_epoch)
        self.epoch_precision_list.append(selected_epoch)
        self.epoch_recall_list.append(selected_epoch)
        self.epoch_f1_list.append(selected_epoch)

        self.train_accs.append(best_train_acc) ; self.valid_accs.append(best_val_acc) ; self.test_accs.append(best_test_acc)
        self.train_precisions.append(best_train_precision) ; self.valid_precisions.append(best_val_precision) ; self.test_precisions.append(best_test_precison)
        self.train_recalls.append(best_train_recall) ; self.valid_recalls.append(best_val_recall) ; self.test_recalls.append(best_test_recall)
        self.train_f1s.append(best_train_f1) ; self.valid_f1s.append(best_val_f1) ; self.test_f1s.append(best_test_f1)

        if fold+1 != self.args.folds:
            self.running_train_accs = [] ; self.running_valid_accs = [] ; self.running_test_accs = []
            self.running_train_precisions = [] ; self.running_valid_precisions = [] ; self.running_test_precisions = []
            self.running_train_recalls = [] ; self.running_valid_recalls = [] ; self.running_test_recalls = []
            self.running_train_f1s = [] ; self.running_valid_f1s = [] ; self.running_test_f1s = []
            
            self.running_decision_thresholds = []
            self.cnt = 0
            self.best_val = 0

            self.cnt_precision=0
            self.best_val_precision=0

            self.cnt_recall=0
            self.best_val_recall=0

            self.cnt_f1=0
            self.best_val_f1=0


    def summary(self):

        train_acc_mean = torch.tensor(self.train_accs).mean().item()
        val_acc_mean = torch.tensor(self.valid_accs).mean().item()
        test_acc_mean = torch.tensor(self.test_accs).mean().item()
        
        train_precision_mean = torch.tensor(self.train_precisions).mean().item()
        val_precision_mean = torch.tensor(self.valid_precisions).mean().item()
        test_precision_mean = torch.tensor(self.test_precisions).mean().item()

        train_recall_mean = torch.tensor(self.train_recalls).mean().item()
        val_recall_mean = torch.tensor(self.valid_recalls).mean().item()
        test_recall_mean = torch.tensor(self.test_recalls).mean().item()

        train_f1_mean = torch.tensor(self.train_f1s).mean().item()
        val_f1_mean = torch.tensor(self.valid_f1s).mean().item()
        test_f1_mean = torch.tensor(self.test_f1s).mean().item()
        
        print("** | train acc : {:.2f} | valid acc : {:.2f} | test acc : {:.2f} |  **\n".format(
                    train_acc_mean, val_acc_mean, test_acc_mean))
        print("** | train precision : {:.2f} | valid precision : {:.2f} | test precision : {:.2f} |  **\n".format(
                    train_precision_mean, val_precision_mean, test_precision_mean))
        print("** | train recall : {:.2f} | valid recall : {:.2f} | test recall : {:.2f} |  **\n".format(
                    train_recall_mean, val_recall_mean, test_recall_mean))
        print("** | train f1 : {:.2f} | valid f1 : {:.2f} | test f1 : {:.2f} |  **\n".format(
                    train_f1_mean, val_f1_mean, test_f1_mean))
        return (
            train_acc_mean, val_acc_mean, test_acc_mean,
            train_precision_mean, val_precision_mean, test_precision_mean,
            train_recall_mean, val_recall_mean, test_recall_mean,
            train_f1_mean, val_f1_mean, test_f1_mean,
        )



            
