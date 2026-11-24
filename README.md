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

Obtain research access to the full TwiBot-20 account records from the dataset authors and use authorized local copies of the rewritten snapshot. Obtain the Cresci-15 conversion from the TwiBot-22 collection. [TwiBot-20 access](https://github.com/BunsenFeng/TwiBot-20#full-twibot-20-dataset) · [Cresci-15 access](https://github.com/LuoUndergradXJTU/TwiBot-22#how-to-download-other-datasets)

| Dataset | Input directory | Required files |
| --- | --- | --- |
| [TwiBot-20](#run-on-twibot-20) | `data/raw/` | `train.json`, `dev.json`, `test.json`. `support.json` is required with `--include-support`. |
| [Cresci-15](#run-on-cresci-15) | Existing external `cresci-2015` directory | `node.json`, `edge.csv`, `label.csv`, `split.csv` |
| [LLM-rewritten TwiBot-20](#run-on-the-llm-rewritten-subset) | `rewrite-dataset/` at the repository root | `train.json`, `dev.json`, `test.json` |

Create `data/raw/` if you use that location for TwiBot-20. All three commands accept an external directory via `--raw-dir`. Run preprocessing first, then pass the generated `.pt` file to `train.py`. Git ignores raw records and generated tensors.

## Run on TwiBot-20

```bash
python -m src.feature_preprocessing --raw-dir data/raw --include-support --output data/twibot20.pt
python train.py --data-file data/twibot20.pt --use-original-splits --label-rate 90 --mode bo --mc-dropout --seed 0 --split-seed 0 --bo-seed 0 --result-json runs/twibot20-seed0.json
```

`--use-original-splits` keeps the original dev and test splits and samples visible labels within each class in the original training split. The remaining training accounts and support accounts are eligible for pseudo-labels. The fixed dev and test accounts stay in the graph for message passing and evaluation. Omitting the switch produces a different split and a transductive candidate pool that can include validation and test nodes.

## Run on Cresci-15

Obtain the converted `cresci-2015` dataset in TwiBot-22 format. Pass the existing directory containing `node.json`, `edge.csv`, `label.csv`, and `split.csv` to `--raw-dir`. The directory can be outside the repository. Run these commands from the repository root. [Cresci-15 download instructions via TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22#how-to-download-other-datasets)

```bash
python -m src.feature_preprocessing --dataset cresci-15 --raw-dir "/path/to/cresci-2015" --output data/cresci15.pt
python train.py --data-file data/cresci15.pt --use-original-splits --label-rate 90 --mode bo --mc-dropout --seed 0 --split-seed 0 --bo-seed 0 --result-json runs/cresci15-seed0.json
```

The preprocessor maps `human` to 0 and `bot` to 1, retains directed `follow` and `friend` account edges as types 0 and 1, and uses up to 20 nonempty tweets per account in the order of the `post` records. The default embedding limit is 128 tokens per text. With `--use-original-splits`, training uses the splits in `split.csv` and samples visible labels within the original training split. The reference copy contains 3,708 training, 1,058 validation, and 535 test accounts.

## Run on the LLM-rewritten subset

Download the [LLM-rewritten TwiBot-20 dataset v1 (ZIP)](https://github.com/LQ4844/LiProS/releases/download/dataset-v1/LiProS-rewrite-dataset-local.zip). Extract `dataset_rewrite/` from the archive and place it at the repository root as `rewrite-dataset/`, with `train.json`, `dev.json`, and `test.json` directly inside it.

The final snapshot has 7,096 accounts and 310,934 changed bot tweets. The command below keeps the 4,967 / 1,419 / 710 train/dev/test assignment and samples visible labels from train.

```bash
python -m src.feature_preprocessing --raw-dir rewrite-dataset --output data/twibot20_rewrite.pt
python train.py --data-file data/twibot20_rewrite.pt --use-original-splits --label-rate 90 --mode bo --mc-dropout --seed 0 --split-seed 0 --bo-seed 0 --result-json runs/rewrite-seed0.json
```

## Encoders and dropout

Select an encoder with `train.py --encoder`. [Encoder implementations](layers/GNN.py)

| Component | CLI option | Role |
| --- | --- | --- |
| `LiProSEncoder` | `--encoder lipros` (default) | Applies attention-based feature fusion and relational graph convolution. |
| `ESAConsEncoder` | `--encoder esa_cons` | Uses `LiProSEncoder` for classification and `SARGCN` for clustering. `--consist-lambda` weights the loss that aligns their normalized embeddings. |
| `SARGCN` | `--encoder sargcn` | Concatenates the projected bio, tweet, numeric profile, and categorical profile streams and applies relational graph convolution using edge types. |
| `FeatureGAT` | `--encoder gat` | Uses the same four profile streams and applies two graph attention layers over the account graph without edge types. |

Set `esa_dropout` with `--dropout` or `--esa-dropout`. The default is `0.4`. [CLI arguments](src/argument.py)

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
| Loss and regularization | Graph and self-loop cross-entropies have equal weight. AdamW uses learning rate 0.001 and `--decay 0.005`. The explicit L2 term λ in the manuscript equation has no separate coefficient in the release. The optional `esa_cons` encoder uses `--consist-lambda 0.1` for its consistency loss. |
| Stages and epochs | `--stage 8` self-training rounds, `--pretrain_epochs 200` per round, `--epochs 1000` maximum final epochs, `--patience 200` based on validation accuracy. |
| Clustering | `--num_K 200`, K-means on L2-normalized embeddings with 10 initializations. |
| Uncertainty | `--mc-dropout` uses 20 stochastic passes. The predicted-class probability standard deviation passes at or below `--kappa-p` (default `4e-8`). |
| Confidence | Single mode defaults to 0.65. BO searches `[0.5, 0.9]` and maximizes validation F1 by default. |
| BO surrogate | Gaussian process with a constant kernel times Matérn-5/2 plus white noise. Initial length scale 0.1, signal variance 1, noise variance 10⁻⁵. The search evaluates 3 initial and 7 adaptive thresholds, scoring 512 candidates per adaptive iteration by expected improvement. |
| Epoch selection | Highest validation accuracy plus validation F1. The first epoch wins ties. Attention-score filtering and decision-threshold calibration default to off. |

## Evaluation and reproducibility

- `--seed` controls model initialization and training randomness. `--split-seed` controls splits or visible training-label sampling. `--bo-seed` controls the Bayesian search. The result JSON records visible split sizes and mask hashes for each fold. The current split generator can produce different masks from earlier releases at the same seed.
- BO fits a Gaussian-process surrogate with a Matérn-5/2 kernel and white noise. It maximizes expected improvement over 512 sampled candidates per adaptive step and evaluates validation F1 at 3 initial thresholds and 7 adaptive thresholds in `[0.5, 0.9]`.
- The CLI selects the earliest epoch with the highest validation F1 plus validation accuracy. Early stopping tracks validation accuracy. Attention-score pseudo-label filtering and per-epoch decision-threshold calibration are off by default because the manuscript describes confidence, uncertainty, and cluster checks with the standard classifier decision. Either optional experiment can be enabled explicitly. Calibration uses the same validation set to fit the threshold and select the epoch.
- `neiBot_ratio` and `neiLabel_ratio` use only original visible training labels, excluding validation, test, and pseudo-labels.
- Use `--device -1` for CPU and `--num-threads N` to set PyTorch CPU threads. The CLI's `--deterministic-algorithms` sets a startup `PYTHONHASHSEED` if unset, configures cuBLAS before importing PyTorch, and warns for unsupported deterministic operations. cuDNN benchmarking is disabled.
- Add `--epoch-csv runs/epochs.csv` for per-epoch pretraining and final training records. Add `--checkpoint-dir checkpoints/my-run` to save the selected weights for each BO trial and fold.

For five independent training runs with a fixed split and BO seed, use model seeds 0–4. For example, after generating `data/twibot20.pt`:

```bash
for seed in 0 1 2 3 4; do
  python train.py --data-file data/twibot20.pt --use-original-splits --label-rate 90 --mode bo --mc-dropout --seed "$seed" --split-seed 0 --bo-seed 0 --result-json "runs/twibot20-seed${seed}.json"
done
python scripts/aggregate_results.py "runs/twibot20-seed*.json"
```

The aggregation script reports the mean and sample standard deviation across matched runs.

## License and acknowledgements

The MIT license covers original LiProS code and documentation. The third-party notices document code provenance and redistribution status for the encoders, classifier, training workflow, and metric helpers. [LiProS MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

BotRGCN is a reference for the text and profile streams and the follow graph. [BotRGCN MIT license](LICENSE-BotRGCN.txt)

Graph convolution and attention operators use PyTorch Geometric. Dataset access and redistribution follow the providers' terms. [Dataset setup](#dataset-setup)
