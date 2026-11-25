# LiProS

LiProS detects social bots with lifecycle features and semi-supervised graph learning. It groups the 41 feature entries in V27 Tables 1–5 into creation, production, dissemination, interaction, and operation. Self-training screens predicted labels by confidence, Monte Carlo dropout uncertainty, and agreement with the cluster class.

## Install

Use Python 3.11 in a fresh environment. First install a PyTorch 2.x build (2.6 or newer) for your CPU or CUDA platform. Install the remaining requirements after confirming the selected build. PyTorch Geometric uses the installed PyTorch build. Match any optional PyG acceleration wheels to that build and CUDA version. [PyTorch installation](https://pytorch.org/get-started/locally/) · [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m pip check
python train.py --help
python -m src.feature_preprocessing --help
```

In an environment where PyG's dependencies are already installed, `python -m pip install --no-deps "torch-geometric>=2.6,<3"` can be used to install PyG separately. The requirements list declares the project's direct packages, including the pinned regex version.

## Dataset setup

Obtain research access to the full TwiBot-20 account records from the dataset authors and use local copies of the rewritten snapshot. Obtain the Cresci-15 conversion from the TwiBot-22 collection. [TwiBot-20 access](https://github.com/BunsenFeng/TwiBot-20#full-twibot-20-dataset) · [Cresci-15 access](https://github.com/LuoUndergradXJTU/TwiBot-22#how-to-download-other-datasets)

| Dataset | Input directory | Required files |
| --- | --- | --- |
| [TwiBot-20](#run-on-twibot-20) | `data/raw/` | `train.json`, `dev.json`, `test.json`. `support.json` is required with `--include-support`. |
| [Cresci-15](#run-on-cresci-15) | Existing external `cresci-2015` directory | `node.json`, `edge.csv`, `label.csv`, `split.csv` |
| [LLM-rewritten TwiBot-20](#run-on-the-llm-rewritten-subset) | `rewrite-dataset/` at the repository root | `train.json`, `dev.json`, `test.json` |

Create `data/raw/` if you use that location for TwiBot-20. All three commands accept an external directory via `--raw-dir`. Run preprocessing first, then pass the generated `.pt` file to `train.py`. Git ignores raw records and generated tensors.

## Run on TwiBot-20

```bash
python -m src.feature_preprocessing --raw-dir data/raw --include-support --output data/twibot20.pt
python train.py --data-file data/twibot20.pt --split-mode random --pseudo-pool vu --label-rate 90 --mode bo --mc-dropout --seed 0 --split-seed 0 --bo-seed 0 --result-json runs/twibot20-seed0.json
```

`--split-mode random` follows Section 4.1 of the paper. It samples the specified percentage from all labeled nodes as visible training labels and rounds the training count down. It assigns 10% of the remaining labeled nodes to validation, rounded down, and the rest to testing. At a 90% label rate, TwiBot-20 has 10,643 visible training nodes, 118 validation nodes, and 1,065 test nodes. Unlabeled support nodes stay in the graph and are eligible for pseudo-labeling.

`--split-mode auto` is the default and selects `random` for TwiBot-20. `--split-mode official` keeps the original dev and test splits and samples visible labels by class within the original training split. `--use-original-splits` is an alias for `--split-mode official`.

`--pseudo-pool vu` is the default and excludes the current fold's validation and test nodes from pseudo-labeling. Those nodes stay in the graph for message passing and evaluation. If the three splits cover every node, the initial candidate pool is empty. `--pseudo-pool all` also allows validation and test nodes to become candidates.

## Run on Cresci-15

Obtain the converted `cresci-2015` dataset in TwiBot-22 format. Pass the existing directory containing `node.json`, `edge.csv`, `label.csv`, and `split.csv` to `--raw-dir`. The directory can be outside the repository. Run these commands from the repository root. [Cresci-15 download instructions via TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22#how-to-download-other-datasets)

```bash
python -m src.feature_preprocessing --dataset cresci-15 --raw-dir "/path/to/cresci-2015" --output data/cresci15.pt
python train.py --data-file data/cresci15.pt --split-mode random --holdout-support 530 --pseudo-pool vu --label-rate 90 --mode bo --mc-dropout --seed 0 --split-seed 0 --bo-seed 0 --result-json runs/cresci15-seed0.json
```

The preprocessor maps `human` to 0 and `bot` to 1, retains directed `follow` and `friend` account edges as types 0 and 1, and uses up to 20 nonempty tweets per account in the order of the `post` records. The default embedding limit is 128 tokens per text.

The Cresci-15 conversion has labels for every account. Following the TwiBot-20 protocol, the command uses a separate support pool. It reserves 530 accounts at random using `--split-seed` and hides their labels at training time. These accounts stay in the graph and are eligible for pseudo-labeling. The random split applies the 90% label rate to the remaining 4,771 accounts, giving 4,293 visible training, 47 validation, and 431 test accounts. The same feature file can be used for runs with or without held-out support.

For the fixed splits from `split.csv`, omit `--holdout-support` and pass `--split-mode official`. The default `auto` mode also selects `official` for Cresci-15 with split metadata. The reference copy contains 3,708 training, 1,058 validation, and 535 test accounts.

## Run on the LLM-rewritten subset

Download the [LLM-rewritten TwiBot-20 dataset v1 (ZIP)](assets/twibot20-rewrite-v1.zip). Extract `dataset_rewrite/` from the archive and place it at the repository root as `rewrite-dataset/`, with `train.json`, `dev.json`, and `test.json` directly inside it.

The final snapshot has 7,096 accounts and 310,934 changed bot tweets. The command below keeps the 4,967 / 1,419 / 710 train/dev/test assignment and samples visible labels from train.

```bash
python -m src.feature_preprocessing --raw-dir rewrite-dataset --output data/twibot20_rewrite.pt
python train.py --data-file data/twibot20_rewrite.pt --split-mode official --pseudo-pool vu --label-rate 90 --mode bo --mc-dropout --seed 0 --split-seed 0 --bo-seed 0 --result-json runs/rewrite-seed0.json
```

## Encoders and dropout

Select an encoder with `train.py --encoder`. [Encoder implementations](layers/GNN.py)

| Component | CLI option | Role |
| --- | --- | --- |
| `LiProSEncoder` | `--encoder lipros` (default) | Applies attention-based feature fusion and relational graph convolution. |
| `ESAConsEncoder` | `--encoder esa_cons` | Uses `LiProSEncoder` for classification and `SARGCN` for clustering. `--consist-lambda` weights the loss that aligns their normalized embeddings. |
| `SARGCN` | `--encoder sargcn` | Concatenates the projected bio, tweet, numeric profile, and categorical profile streams and applies relational graph convolution using edge types. |
| `FeatureGAT` | `--encoder gat` | Uses the same four profile streams and applies two graph attention layers over the account graph without edge types. |

The default dropout probability is `0.4` for training and for stochastic inference when `--mc-dropout` is enabled. Set it with `--dropout` or `--esa-dropout`. [CLI arguments](src/argument.py)

## Feature mapping and parameters

### Lifecycle grouping

V27 Tables 1–5 and Fig. 1(a) organize the features into five groups by behavioral function. The 41 entries comprise 39 scalar features and two text embeddings. [Feature definitions](src/dataset.py)

| Lifecycle stage | Entries | Features |
| --- | ---: | --- |
| Creation | 8 | `screenName_len`, `is_verified`, `is_protected`, `Is_geo_enabled`, `use_customBg`, `bgTile_enabled`, `is_defProfile`, `is_defAvatar` |
| Production | 8 | `tweet_Emb`, `Desc_Emb`, `tweetLenMax_medRatio`, `emoji_repeatRate`, `emojiType_perTweet`, `topEmoji_ratio`, `emojiAvg_perTweet`, `emojiUse_ratio` |
| Dissemination | 10 | `senti_consistency`, `senti_volatility`, `senti_complex`, `senti_spanDiff`, `senti_pos`, `senti_neg`, `senti_neu`, `bio_promoWord`, `useURL_ratio`, `bio_hasURL` |
| Interaction | 8 | `cluster_coef`, `deg_centrality`, `neiBot_ratio`, `neiLabel_ratio`, `follower_cnt`, `follow_cnt`, `like_cnt`, `reply_count` |
| Operation | 7 | `bioTopic`, `post_frequency`, `totalTweet_cnt`, `is_translator`, `contributor_enabled`, `translation_enabled`, `has_extProfile` |

The dissemination stage uses cosine similarity between adjacent VADER positive, negative, and neutral vectors. `senti_spanDiff` sums the maximum positive and maximum negative scores across an account's tweets.

Rebuild feature files with `python -m src.feature_preprocessing` to use the v3 feature set. Existing v1/v2 files remain readable.

`reply_count` measures lifetime replies posted by an account. Supply an optional `--reply-counts accounts.csv` with `id,reply_count` columns, matching TwiBot-20 `ID` or Cresci-15 `node.json` account IDs. A CSV entry takes precedence over the account-level `reply_count` and then the profile field of the same name. Observed counts use `log(1 + reply_count)`. Missing counts use `-1`. The preprocessor logs the number of accounts with observed counts. Text-only tweet records provide no lifetime count.

`bioTopic` uses a fixed English keyword vocabulary because V27 leaves the category definitions open. It chooses the category with the most distinct keyword matches after removing URLs and mentions. Ties use category order: technology (2), business (3), sports (4), arts (5), news and politics (6), education and science (7), lifestyle (8). Empty bios map to 0 and unmatched bios to 1.

`neiBot_ratio` divides visible bot neighbors by all visibly labeled neighbors. `neiLabel_ratio` divides visible human neighbors by all distinct neighbors. Both numerators use exclusively the fold's original visible ground-truth training labels. Validation labels, test labels, pseudo-labels, and support labels are excluded. Both columns start at zero in the feature file and are recomputed for each fold.

### Training parameters

| Parameter | Release value and flag |
| --- | --- |
| Projection and hidden size | `--fusion-dim 256`, `--layers '[128, 128]'`. The default shared RGCN performs two passes. |
| Dropout | See [Encoders and dropout](#encoders-and-dropout). |
| Loss and regularization | Graph and self-loop cross-entropies have equal weight. The loss adds the sum of squared model parameters scaled by `--l2-lambda 1e-5`. AdamW separately uses learning rate 0.001 and weight decay `--decay 0.005`. The optional `esa_cons` encoder uses `--consist-lambda 0.1` for its consistency loss. |
| Self-training budget | Both modes default to up to 200 rounds (`--stage 200`) of 200 pretraining epochs each (`--pretrain_epochs 200`). Model training continues between rounds by default. Pass `--ensemble-reinit-each-stage` to reinitialize the model between rounds. |
| Candidate threshold | Self-training stops when the current unlabeled candidate count falls below `--min-unlabeled` (default 1). With clustering enabled, each round also requires at least `--num_K` candidates. Final training runs even when the initial candidate pool is empty. |
| Final training | Up to 1,000 epochs per fold and trial, with early stopping after 200 epochs without an improvement in validation accuracy (`--patience 200`). |
| Clustering | `--num_K 200`. K-means fits the L2-normalized embeddings of the current unlabeled candidates with 10 initializations. Class centers are computed from the current training set. |
| Uncertainty | `--mc-dropout` uses 20 stochastic passes. The predicted-class probability standard deviation passes at or below `--kappa-p` (default `0.3`). |
| Confidence | Single mode defaults to 0.65. BO searches `[0.5, 0.9]` and maximizes validation F1 by default. |
| BO surrogate | Gaussian process with a constant kernel times Matérn-5/2 plus white noise. Initial length scale 0.1, signal variance 1, noise variance 10⁻⁵. The default search evaluates 3 initial and 2 adaptive thresholds, scoring 512 candidates per adaptive iteration by expected improvement. |
| Epoch selection | Highest validation accuracy plus validation F1. The first epoch wins ties. Attention-score filtering and decision-threshold calibration default to off. |

## Evaluation and reproducibility

- `--seed` controls model initialization and training randomness. `--split-seed` controls splits or visible training-label sampling. `--bo-seed` controls the Bayesian search. The result JSON records visible split sizes and mask hashes for each fold. The current split generator can produce different masks from earlier releases at the same seed.
- The BO evaluation budget is set by `--bo-initial-points` and `--bo-iterations`. The defaults give 5 evaluations per run.
- The CLI selects the earliest epoch with the highest validation F1 plus validation accuracy. Early stopping tracks validation accuracy. Attention-score pseudo-label filtering and per-epoch decision-threshold calibration are off by default because the manuscript describes confidence, uncertainty, and cluster checks with the standard classifier decision. Either optional experiment can be enabled explicitly. Calibration uses the same validation set to fit the threshold and select the epoch.
- `neiBot_ratio` and `neiLabel_ratio` use only original visible training labels, excluding validation, test, and pseudo-labels.
- Use `--device -1` for CPU and `--num-threads N` to set PyTorch CPU threads. The CLI's `--deterministic-algorithms` sets a startup `PYTHONHASHSEED` if unset, configures cuBLAS before importing PyTorch, and warns for unsupported deterministic operations. cuDNN benchmarking is disabled.
- Add `--epoch-csv runs/epochs.csv` for per-epoch pretraining and final training records. Add `--checkpoint-dir checkpoints/my-run` to save the selected weights for each BO trial and fold.

For five independent training runs with a fixed split and BO seed, use model seeds 0–4. For example, after generating `data/twibot20.pt`:

```bash
for seed in 0 1 2 3 4; do
  python train.py --data-file data/twibot20.pt --split-mode random --pseudo-pool vu --label-rate 90 --mode bo --mc-dropout --seed "$seed" --split-seed 0 --bo-seed 0 --result-json "runs/twibot20-seed${seed}.json"
done
python scripts/aggregate_results.py "runs/twibot20-seed*.json"
```

The aggregation script reports the mean and sample standard deviation across matched runs.


