# LiProS

LiProS detects social bots with lifecycle features and semi-supervised graph learning. The five input streams are bio embeddings, tweet embeddings, numeric and categorical profile attributes, and 37 auxiliary lifecycle columns. A single attention head and a mean residual fuse the projected inputs before two RGCN passes. Self-training screens predicted labels by confidence, Monte Carlo dropout uncertainty, and agreement with the cluster class.

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

## Feature mapping and parameters

### Input streams

The paper groups features by five lifecycle stages. The release projects five input streams and combines them with attention and a mean residual.

| Stream | Width | Contents |
| --- | ---: | --- |
| Bio embedding | 768 by default | Table 2 `Desc_Emb`. Pooled vectors from `distilroberta-base`. |
| Tweet embedding | 768 by default | Table 2 `tweet_Emb`. `roberta-base` token vectors pooled per tweet and then per account. |
| Numeric profile | 6 | Followers, following, favorites, statuses, screen-name length, and account age. |
| Categorical profile | 11 | Protected, geo enabled, verified, contributors enabled, translator, translation enabled, background tile, use background image, extended profile, default profile, and default profile image. |
| Auxiliary lifecycle | **37** | The five groups listed below, concatenated before projection. |

| Lifecycle group | Width | Auxiliary columns in order |
| --- | ---: | --- |
| Creation | 8 | Screen-name length, verified, protected, geo enabled, use background image, background tile, default profile, default profile image |
| Production | 6 | Maximum-to-median tweet length, emoji repetition rate, mean distinct emoji per tweet, most frequent emoji proportion, mean emoji count, fraction of tweets with emoji |
| Dissemination | 10 | Adjacent-tweet sentiment consistency, standard deviation of sentiment entropy, mean sentiment entropy, absolute difference of maximum positive and negative sentiment, mean positive, mean negative, mean neutral, promotional terms in bio, tweet URL fraction, bio URL flag |
| Interaction | 7 | Local clustering, normalized degree, `neiBot_ratio`, `neiLabel_ratio`, followers, following, favorites |
| Operation | 6 | Statuses per day, status count, translator, contributors enabled, translation enabled, extended profile |

Manuscript Tables 1–5 list 41 features. The two embeddings use separate streams, and `reply_count` and `bioTopic` are absent from this preprocessor, leaving **8 + 6 + 10 + 7 + 6 = 37** auxiliary columns. Several account attributes also appear in the numeric and categorical streams. The released implementation uses the absolute difference of the positive and negative sentiment peaks.

`neiBot_ratio` divides visible bot neighbors by all visibly labeled neighbors. `neiLabel_ratio` divides visible human neighbors by all distinct neighbors. Both numerators use exclusively the fold's original visible ground-truth training labels. Validation labels, test labels, pseudo-labels, and support labels are excluded. Both columns start at zero in the feature file and are recomputed for each fold.

### Training parameters

| Parameter | Release value and flag |
| --- | --- |
| Projection and hidden size | `--fusion-dim 256`, `--layers '[128, 128]'`. The default shared RGCN performs two passes. |
| Dropout | `--dropout 0.4`. |
| Loss and regularization | Graph and self-loop cross-entropies have equal weight. AdamW uses learning rate 0.001 and `--decay 0.005`. The explicit L2 term λ in the manuscript equation has no separate coefficient in the release. The optional `esa_cons` encoder uses `--consist-lambda 0.1` for its consistency loss. |
| Stages and epochs | `--stage 8` self-training rounds, `--pretrain_epochs 200` per round, `--epochs 1000` maximum final epochs, `--patience 200` based on validation accuracy. |
| Clustering | `--num_K 200`, K-means on L2-normalized embeddings with 10 initializations. |
| Uncertainty | `--mc-dropout` uses 20 stochastic passes. `--kappa-p 4e-8` sets the predicted-class probability standard deviation cutoff. |
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

The root MIT license covers original LiProS material. Retained upstream code follows its corresponding license terms. [LiProS MIT license](LICENSE)

| Upstream | Use in LiProS | License and status |
| --- | --- | --- |
| ESA-BotRGCN | Multistream fusion and experimental encoders in `layers/GNN.py`. | MIT, Copyright (c) 2025 filipovicSamuel. [License text](LICENSE-ESA-BotRGCN.txt) · [Paper](https://doi.org/10.3390/s25134179) |
| BotRGCN | Reference for the text and profile streams and follow graph in `src/feature_preprocessing.py` and `layers/GNN.py`. | MIT, Copyright (c) 2021 wind_binteng. [License text](LICENSE-BotRGCN.txt) · [Upstream license](https://github.com/BunsenFeng/BotRGCN/blob/main/LICENSE) |
| M3S PyTorch implementation | Multi-stage self-training in `models/lipros.py`, the classifier in `layers/Classifier.py`, SARGCN, RGCN, and HGT in `layers/GNN.py`, and training and metric helpers in `src/utils.py`. | No project license is listed upstream. [Upstream repository](https://github.com/Junseok0207/M3S_Pytorch) · [Paper](https://arxiv.org/abs/1902.11038) |

GCN, GAT, and HGT operators use PyTorch Geometric. Dataset access and redistribution follow the providers' terms. [Dataset setup](#dataset-setup)
