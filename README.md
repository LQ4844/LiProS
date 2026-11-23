# LiProS

LiProS is a semi-supervised method for social bot detection. The feature preprocessor computes account statistics across five lifecycle stages spanning creation, production, dissemination, interaction, and operation. The encoder projects its five input streams into a shared 256-dimensional space and combines them using single-head attention and a mean residual connection. A two-layer relational graph convolutional network (RGCN) then aggregates outgoing and incoming follow relations separately. During self-training, candidate pseudo-labels are screened in sequence using a confidence threshold tuned by Bayesian optimization on validation F1, uncertainty estimated with Monte Carlo dropout, and agreement with the class assigned to each cluster.

## Datasets

| Dataset | GitHub repository | Download | Files |
| --- | --- | --- | --- |
| TwiBot-20 | [TwiBot-20 repository](https://github.com/BunsenFeng/TwiBot-20) | [Official access and download instructions](https://github.com/BunsenFeng/TwiBot-20#full-twibot-20-dataset) | `train.json`, `dev.json`, `test.json` |
| Cresci-15 | [TwiBot-22 repository](https://github.com/LuoUndergradXJTU/TwiBot-22#how-to-download-other-datasets) | [TwiBot-22 dataset download folder](https://drive.google.com/drive/folders/1gXFZp3m7TTU-wyZRUiLHdf_sIZpISrze?usp=sharing) | `cresci-2015` in the converted datasets archive |

The paper evaluates LiProS on TwiBot-20 and Cresci-15. The preprocessing script in this release reads TwiBot-20 JSON files. The Cresci-15 download contains the dataset converted to the TwiBot-22 format.

## Run LiProS

Use Python 3.11 and [install PyTorch for your CPU or CUDA setup](https://pytorch.org/get-started/locally/). Place the three labeled TwiBot-20 JSON files under `data/raw/` and run the following commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m src.feature_preprocessing --raw-dir data/raw --output data/twibot20.pt
python train.py --data-file data/twibot20.pt --label-rate 90 --mode bo --mc-dropout --no-use-attention-filter --no-calibrate-decision-threshold --result-json runs/twibot20_seed0.json
```

The five LiProS input streams are account bio embeddings, tweet embeddings, numeric profile attributes, categorical profile attributes, and 37 auxiliary lifecycle features. The two neighbor-label columns are computed for each fold from its original training labels. Existing LiProS `.pt` feature files can be reused.

These commands use the default random seeds, a 90% training label rate, Bayesian threshold search, Monte Carlo dropout, and cluster consistency checks. By default, `--mode bo` searches the pseudo-label confidence threshold range `[0.5, 0.9]` and selects the threshold with the highest validation F1.

## Supplementary LLM Rewrite Experiment

The rewrite experiment uses a stratified sample of 7,096 TwiBot-20 accounts and preserves their original `train`, `dev`, and `test` assignments. See the [dataset summary](dataset_summary.csv) for counts by split. The full rewritten dataset contains original account profiles, relationships, and identifiable text. Exact reproduction at the tweet level requires an authorized local copy of the full rewritten dataset.

To run LiProS on the rewritten data, place the JSON files under `data/dataset_rewrite/` and run the following commands. The training script creates new splits. The supplementary materials document the evaluation protocol that uses the original fixed splits.

```bash
python -m src.feature_preprocessing --raw-dir data/dataset_rewrite --output data/twibot20_rewrite.pt
python train.py --data-file data/twibot20_rewrite.pt --label-rate 90 --mode bo --mc-dropout --no-use-attention-filter --no-calibrate-decision-threshold --result-json runs/rewrite_seed0.json
```

## Supplementary Materials

- [LLM Rewrite and Baseline Evaluation Parameters](docs/llm-rewrite-experiment.md)
- [LiProS Experiment Configuration and Parameters](docs/lipros-experiment-appendix.md)
