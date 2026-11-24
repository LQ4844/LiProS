import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import networkx as nx
import numpy as np
import regex
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.dataset import DATASET_VERSION, LABELED_SPLITS, STATIC_WIDTHS, UNLABELED, load_dataset


PROFILE_FLAGS = (
    "protected", "geo_enabled", "verified", "contributors_enabled",
    "is_translator", "is_translation_enabled", "profile_background_tile",
    "profile_use_background_image", "has_extended_profile", "default_profile",
    "default_profile_image",
)
PROFILE_COUNTS = ("followers_count", "friends_count", "favourites_count", "statuses_count")

STAGES = tuple(STATIC_WIDTHS)
INTERACTION_OFFSET = STATIC_WIDTHS["creation"] + STATIC_WIDTHS["production"] + STATIC_WIDTHS["dissemination"]
NEIGHBOR_LABEL_COLUMNS = slice(INTERACTION_OFFSET + 2, INTERACTION_OFFSET + 4)
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
PROMO = re.compile(r"\b(?:buy|shop|sale|discount|deal|promo|free|win|giveaway|subscribe|"
                   r"follow\s+me|click\s+here|link\s+in\s+bio)\b", re.I)
BIO_TOPIC_WORDS = {
    "technology": {"ai", "coding", "developer", "programmer", "programming", "software", "tech", "technology"},
    "business": {"business", "entrepreneur", "finance", "investing", "marketing", "startup", "trading"},
    "sports": {"athlete", "baseball", "basketball", "football", "soccer", "sports", "tennis"},
    "arts": {"art", "artist", "film", "music", "musician", "photographer", "photography", "poetry"},
    "news_politics": {"elections", "government", "journalism", "journalist", "news", "politics"},
    "education_science": {"education", "professor", "research", "researcher", "science", "teacher", "university"},
    "lifestyle": {"cooking", "fashion", "fitness", "food", "health", "travel", "wellness", "yoga"},
}
GRAPHEME = regex.compile(r"\X")
PICTOGRAPH = regex.compile(r"\p{Extended_Pictographic}")
SNAPSHOT = datetime(2020, 9, 1, tzinfo=timezone.utc)
CRESCI_SNAPSHOT = datetime(2015, 10, 1, tzinfo=timezone.utc)
CRESCI_MAX_TWEETS = 20
CRESCI_SPLITS = {"train": "train", "val": "dev", "dev": "dev", "test": "test"}
CRESCI_LABELS = {"human": 0, "bot": 1}
CRESCI_RELATIONS = {
    "follow": 0, "following": 0, "friend": 1, "friends": 1, "follower": 1, "followers": 1,
}
JSON_WHITESPACE = re.compile(r"[ \t\n\r]*")


def _number(value):
    try:
        return max(float(value or 0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _flag(value):
    return float(str(value).strip().lower() in ("true", "1"))


def _parse_reply_count(value):
    try:
        count = int(str(value).strip())
    except ValueError:
        return None
    return count if count >= 0 else None


def read_reply_counts(path):
    counts = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not {"id", "reply_count"}.issubset(rows.fieldnames or ()):
            raise ValueError(f"{path} must have id and reply_count columns.")
        for row in rows:
            uid = (row["id"] or "").strip()
            count = _parse_reply_count(row["reply_count"])
            if not uid or uid in counts or count is None:
                raise ValueError(f"{path}:{rows.line_num}: expected a unique id and a nonnegative integer reply_count.")
            counts[uid] = count
    return counts


def _reply_feature(user, profile, reply_counts):
    uid = str(user.get("ID", "")).strip()
    for value in (reply_counts.get(uid), user.get("reply_count"), profile.get("reply_count")):
        count = _parse_reply_count(value)
        if count is not None:
            return math.log(count + 1)
    return -1.0  # A known zero replies maps to zero under log(1 + count).


def _bio_topic(bio):
    text = bio.casefold().strip()
    if text in ("", "none", "null", "n/a"):
        return 0
    words = set(re.findall(r"[a-z]+", re.sub(r"@\w+", "", URL.sub("", text))))
    scores = [len(words & keywords) for keywords in BIO_TOPIC_WORDS.values()]
    highest = max(scores)
    return 2 + scores.index(highest) if highest else 1


def _account_age(value, snapshot):
    text = str(value or "").strip()
    try:
        created = datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        try:
            created = datetime.fromisoformat(text)
        except ValueError:
            return 0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (snapshot - created).days


def _emoji(text):
    return [part for part in GRAPHEME.findall(text) if PICTOGRAPH.search(part)]


def _sentiment(tweets, analyzer):
    scores = [analyzer.polarity_scores(t) for t in tweets if t.strip()]
    if not scores:
        return [0.0] * 7
    vectors = np.array([[s["pos"], s["neg"], s["neu"]] for s in scores], dtype=np.float64)
    entropy = -(vectors * np.log(vectors + 1e-12)).sum(axis=1)
    if len(vectors) > 1:
        norms = np.linalg.norm(vectors, axis=1)
        denom = norms[:-1] * norms[1:]
        valid = denom > 0
        consistency = float(np.mean(np.sum(vectors[:-1][valid] * vectors[1:][valid], axis=1)
                                    / denom[valid])) if valid.any() else 0.0
    else:
        consistency = 0.0
    return [consistency, float(entropy.std(ddof=1)) if len(entropy) > 1 else 0.0,
            float(entropy.mean()), float(vectors[:, 0].max() + vectors[:, 1].max()),
            *vectors.mean(axis=0).tolist()]


def _standardize(x, count_columns=()):
    x = np.asarray(x, dtype=np.float32)
    for col in count_columns:
        x[:, col] = np.log1p(x[:, col])
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return torch.from_numpy(np.nan_to_num((x - mean) / np.maximum(std, 1e-6)))


def _graph_features(edge_index, n_users):
    src, dst = edge_index.detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(n_users))
    graph.add_edges_from((int(a), int(b)) for a, b in zip(src, dst, strict=True) if a != b)
    degree = np.array([graph.degree(i) for i in range(n_users)], dtype=np.float32)
    clustering = nx.clustering(graph)
    cluster = np.array([clustering[i] for i in range(n_users)], dtype=np.float32)
    return cluster, degree / max(n_users - 1, 1)


def extract_lifecycle_features(users, edge_index, n_users, snapshot=SNAPSHOT, *, reply_counts=None):
    analyzer = SentimentIntensityAnalyzer()
    rows = {name: [] for name in STAGES}
    reply_counts = reply_counts or {}
    replies, bio_topics = [], []
    for user in users:
        profile = user.get("profile") or {}
        tweets = [t for t in (user.get("tweet") or []) if isinstance(t, str)]
        bio = str(profile.get("description") or "")
        emoji_lists = [_emoji(t) for t in tweets]
        all_emoji = [e for batch in emoji_lists for e in batch]
        freq = Counter(all_emoji)
        n_tweet = len(tweets)
        n_emoji = len(all_emoji)
        lengths = [len(t) for t in tweets]
        median = float(np.median(lengths)) if lengths else 0.0
        rows["creation"].append([
            len(str(profile.get("screen_name") or "").strip()),
            *(_flag(profile.get(key)) for key in (
                "verified", "protected", "geo_enabled", "profile_use_background_image",
                "profile_background_tile", "default_profile", "default_profile_image")),
        ])
        rows["production"].append([
            max(lengths) / median if median else 0.0,
            (n_emoji - len(freq)) / n_emoji if n_emoji else 0.0,
            sum(len(set(batch)) for batch in emoji_lists) / n_tweet if n_tweet else 0.0,
            max(freq.values()) / n_emoji if n_emoji else 0.0,
            n_emoji / n_tweet if n_tweet else 0.0,
            sum(bool(batch) for batch in emoji_lists) / n_tweet if n_tweet else 0.0,
        ])
        rows["dissemination"].append(_sentiment(tweets, analyzer) + [
            len(PROMO.findall(bio)),
            sum(bool(URL.search(t)) for t in tweets) / n_tweet if n_tweet else 0.0,
            float(bool(URL.search(bio))),
        ])
        followers = _number(profile.get("followers_count"))
        following = _number(profile.get("friends_count"))
        likes = _number(profile.get("favourites_count"))
        rows["interactive"].append([followers, following, likes])
        replies.append(_reply_feature(user, profile, reply_counts))
        total = _number(profile.get("statuses_count"))
        age_days = max(_account_age(profile.get("created_at"), snapshot), 1)
        bio_topics.append(_bio_topic(bio))
        rows["operational"].append([
            total / age_days, total,
            *(_flag(profile.get(key)) for key in (
                "is_translator", "contributors_enabled", "is_translation_enabled",
                "has_extended_profile")),
        ])
    cluster, degree = _graph_features(edge_index, n_users)
    interactive = _standardize(rows["interactive"], count_columns=(0, 1, 2))
    stages = {
        "creation": _standardize(rows["creation"]),
        "production": _standardize(rows["production"]),
        "dissemination": _standardize(rows["dissemination"], count_columns=(7,)),
        "interactive": torch.column_stack((torch.from_numpy(cluster), torch.from_numpy(degree),
                                             torch.zeros(n_users, 2), interactive,
                                             torch.tensor(replies, dtype=torch.float32))),
        "operational": torch.column_stack((torch.tensor(bio_topics, dtype=torch.float32),
                                           _standardize(rows["operational"], count_columns=(0, 1)))),
    }
    return stages


def combine_auxiliary_features(stages):
    return torch.cat([stages[name] for name in STAGES], dim=1)


def compute_neighbor_label_features(edge_index, labels, train_mask):
    train = train_mask.detach().cpu().bool().reshape(-1)
    y = labels.detach().cpu().long().reshape(-1)
    n = y.numel()
    if train.numel() != n:
        raise ValueError("The training mask must match the number of labels.")
    source, target = edge_index.detach().cpu().long()
    valid = (source >= 0) & (target >= 0) & (source < n) & (target < n) & (source != target)
    src, dst = source[valid], target[valid]
    pair_ids = torch.unique(torch.cat((src * n + dst, dst * n + src)))
    src, dst = pair_ids // n, pair_ids % n
    degree = torch.bincount(src, minlength=n).float()
    known = train[dst]
    known_count = torch.bincount(src[known], minlength=n).float()
    bot_count = torch.bincount(src[known & (y[dst] == 1)], minlength=n).float()
    human_count = torch.bincount(src[known & (y[dst] == 0)], minlength=n).float()
    return torch.stack((bot_count / known_count.clamp_min(1), human_count / degree.clamp_min(1)), dim=1)


def iter_json_objects(path):
    """Decode an array of objects without keeping the full JSON file in memory."""
    decoder = json.JSONDecoder()
    block_size = 256 * 1024
    with path.open(encoding="utf-8-sig") as handle:
        buffer = handle.read(block_size)
        position = 0

        def next_character():
            nonlocal buffer, position
            while True:
                position = JSON_WHITESPACE.match(buffer, position).end()
                if position < len(buffer):
                    return buffer[position]
                buffer = handle.read(block_size)
                position = 0
                if not buffer:
                    return ""

        if next_character() != "[":
            raise ValueError(f"{path} must contain an array of JSON objects.")
        position += 1
        if next_character() != "]":
            while True:
                if next_character() != "{":
                    raise ValueError(f"{path} must contain only JSON objects.")
                while True:
                    try:
                        row, position = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError as error:
                        block = handle.read(block_size)
                        if not block:
                            raise ValueError(f"Invalid JSON object in {path}") from error
                        buffer = buffer[position:] + block
                        position = 0
                yield row
                delimiter = next_character()
                if delimiter == "]":
                    break
                if delimiter != ",":
                    raise ValueError(f"Expected ',' or ']' after an object in {path}.")
                position += 1
        position += 1
        if next_character():
            raise ValueError(f"Unexpected data after the JSON array in {path}.")


class RawAccounts:
    def __init__(self, raw_dir, include_support=False):
        splits = (*LABELED_SPLITS, "support") if include_support else LABELED_SPLITS
        self.paths = {split: raw_dir / f"{split}.json" for split in splits}
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. See the Dataset setup section of README.md.")

    def __iter__(self):
        for path in self.paths.values():
            yield from iter_json_objects(path)

    def graph_rows(self):
        rows, split_sizes = [], {}
        for split, path in self.paths.items():
            start = len(rows)
            for row in iter_json_objects(path):
                label = UNLABELED if split == "support" else int(row["label"])
                if split != "support" and label not in (0, 1):
                    raise ValueError(f"{path} contains a label outside the binary classes 0 and 1.")
                rows.append({"ID": row["ID"], "neighbor": row.get("neighbor"), "label": label})
            count = len(rows) - start
            if not count:
                raise ValueError(f"{path} must contain at least one account.")
            split_sizes[split] = count
            print(f"[data] {path.name}: {count} accounts", flush=True)
        return rows, split_sizes


def read_cresci15(raw_dir):
    paths = {name: raw_dir / name for name in ("node.json", "edge.csv", "label.csv", "split.csv")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Point --raw-dir to the converted cresci-2015 directory.")

    def csv_rows(name):
        with paths[name].open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)

    labels = {}
    for row in csv_rows("label.csv"):
        uid, label = row["id"].strip(), row["label"].strip().lower()
        if not uid or uid in labels or label not in CRESCI_LABELS:
            raise ValueError("Cresci-15 label.csv must list unique account IDs with human or bot labels.")
        labels[uid] = CRESCI_LABELS[label]

    split_ids = {name: [] for name in LABELED_SPLITS}
    for row in csv_rows("split.csv"):
        uid = row["id"].strip()
        if uid not in labels:
            continue
        split = CRESCI_SPLITS.get(row["split"].strip().lower())
        if split is None:
            raise ValueError(f"Unknown Cresci-15 split: {row['split']!r}. Expected train, val/dev, or test.")
        split_ids[split].append(uid)
    split_sizes = {name: len(ids) for name, ids in split_ids.items()}
    user_ids = [uid for group in split_ids.values() for uid in group]
    user_rows = {uid: i for i, uid in enumerate(user_ids)}
    if len(user_ids) != len(user_rows) or user_rows.keys() != labels.keys() or not all(split_sizes.values()):
        raise ValueError("Cresci-15 split.csv must list every labeled account once across nonempty splits.")

    users = [{"ID": uid, "label": labels[uid], "tweet": []} for uid in user_ids]
    texts = {}
    print("[data] Reading Cresci-15 node.json", flush=True)
    for node in iter_json_objects(paths["node.json"]):
        uid = str(node["id"]).strip()
        if uid in user_rows:
            metrics = node.get("public_metrics") or {}
            users[user_rows[uid]]["profile"] = {
                **node,
                "description": str(node.get("description") or "").strip() or "None",
                "screen_name": node.get("username") or node.get("screen_name"),
                "followers_count": metrics.get("followers_count", node.get("followers_count")),
                "friends_count": metrics.get("following_count", node.get("friends_count")),
                "statuses_count": metrics.get("tweet_count", node.get("statuses_count")),
                "favourites_count": metrics.get("favourites_count", node.get("favourites_count")),
            }
        elif uid.startswith("t"):
            content = node.get("text")
            if isinstance(content, str) and content.strip():
                texts[uid] = content.strip()

    print("[data] Reading Cresci-15 edge.csv", flush=True)
    endpoints, relations = [], []
    for row in csv_rows("edge.csv"):
        source = user_rows.get(row["source_id"].strip())
        if source is None:
            continue
        relation = row["relation"].strip().lower()
        target = row["target_id"].strip()
        if relation == "post":
            tweets = users[source]["tweet"]
            if len(tweets) < CRESCI_MAX_TWEETS and target in texts:
                tweets.append(texts[target])
        elif relation in CRESCI_RELATIONS and target in user_rows:
            endpoints.append((source, user_rows[target]))
            relations.append(CRESCI_RELATIONS[relation])
    print(f"[data] Cresci-15 splits: {split_sizes}", flush=True)
    edges = torch.tensor(endpoints, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return users, split_sizes, edges, torch.tensor(relations, dtype=torch.long)


def profile_features(users, snapshot=SNAPSHOT, *, log_counts=False):
    numeric = []
    categorical = []
    for user in users:
        profile = user.get("profile") or {}
        numeric.append([
            *(float(profile.get(key) or 0) for key in PROFILE_COUNTS),
            len(profile.get("screen_name") or ""),
            _account_age(profile.get("created_at"), snapshot),
        ])
        categorical.append([
            str(profile.get(key)).strip().lower() in ("true", "1") for key in PROFILE_FLAGS
        ])
    values = torch.tensor(numeric, dtype=torch.float32)
    if log_counts:
        columns = [0, 1, 2, 3, 5]
        values[:, columns] = values[:, columns].clamp_min(0).log1p()
    values = (values - values.mean(dim=0)) / values.std(dim=0).clamp_min(1e-8)
    return values, torch.tensor(categorical, dtype=torch.float32)


def follow_graph(users):
    ids = {str(user["ID"]).strip(): row for row, user in enumerate(users)}
    if len(ids) != len(users):
        raise ValueError("Duplicate account IDs across the TwiBot-20 splits.")
    endpoints = []
    relations = []
    for source, user in enumerate(users):
        neighbors = user.get("neighbor") or {}
        for relation, key in enumerate(("following", "follower")):
            for account_id in neighbors.get(key) or []:
                target = ids.get(str(account_id).strip())
                if target is not None:
                    endpoints.append((source, target))
                    relations.append(relation)
    edges = torch.tensor(endpoints, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return edges, torch.tensor(relations, dtype=torch.long)


def bios(users):
    for row, user in enumerate(users):
        description = (user.get("profile") or {}).get("description")
        yield row, str(description) if description is not None else "None"


def tweet_texts(users):
    for row, user in enumerate(users):
        tweets = user.get("tweet")
        if tweets is None:
            yield row, ""
        else:
            if isinstance(tweets, str):
                tweets = [tweets]
            for tweet in tweets:
                if isinstance(tweet, str):
                    yield row, tweet


def encode_users(rows, count, model_name, args):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=args.local_files_only)
    model = model.to(args.device).eval()
    sums = torch.zeros(count, model.config.hidden_size, dtype=torch.float32)
    counts = torch.zeros(count, 1, dtype=torch.float32)
    indices, texts = [], []
    processed = 0

    @torch.inference_mode()
    def add_batch():
        tokens = tokenizer(texts, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt").to(args.device)
        hidden = model(**tokens).last_hidden_state.float()
        mask = tokens.attention_mask.unsqueeze(-1)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).cpu()
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        sums.index_add_(0, indices_tensor, pooled)
        counts.index_add_(0, indices_tensor, torch.ones(len(indices), 1))

    print(f"[embeddings] {model_name} on {args.device}", flush=True)
    for row, text in rows:
        indices.append(row)
        texts.append(text)
        if len(texts) == args.batch_size:
            add_batch()
            processed += len(texts)
            if processed % (args.batch_size * 250) == 0:
                print(f"[embeddings] {processed} texts", flush=True)
            indices.clear()
            texts.clear()
    if texts:
        add_batch()
        processed += len(texts)
    print(f"[embeddings] {processed} texts complete", flush=True)
    return sums / counts.clamp_min(1)


def reuse_embeddings(directory, users, labels):
    def read_tensor(name):
        return torch.load(directory / name, map_location="cpu", weights_only=True)

    if not torch.equal(read_tensor("label.pt").reshape(-1).long(), labels):
        raise ValueError("Legacy label.pt does not match the prepared train + dev + test label sequence.")
    description_path = directory / "description.npy"
    if description_path.is_file():
        old_bios = np.load(description_path, allow_pickle=False)
        if old_bios.tolist() != [text for _, text in bios(users)]:
            raise ValueError("Legacy description.npy does not match raw account order.")
    description, tweets = (read_tensor(name).float() for name in ("des_tensor.pt", "tweets_tensor.pt"))
    return description, tweets


def save_dataset(data, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(data, temporary)
        load_dataset(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate LiProS tensors from TwiBot-20 or converted Cresci-15.")
    parser.add_argument("--dataset", choices=("twibot-20", "cresci-15"), default="twibot-20")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing the raw dataset files.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--description-model", default="distilbert/distilroberta-base")
    parser.add_argument("--tweet-model", default="FacebookAI/roberta-base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-support", action="store_true",
                        help="Also include unlabeled accounts from support.json in the graph and feature tensors.")
    parser.add_argument("--device", default="auto", help="auto, cpu or a CUDA device, e.g. cuda:0")
    parser.add_argument("--num-threads", type=int, help="Intra-op CPU threads. Defaults to PyTorch's setting.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, help="Maximum tokens per text (TwiBot-20: 512, Cresci-15: 128).")
    parser.add_argument("--reuse-embeddings", type=Path,
                        help="Optional directory with matching des_tensor.pt, tweets_tensor.pt and label.pt")
    parser.add_argument("--reply-counts", type=Path, metavar="CSV",
                        help="Optional CSV of lifetime replies posted by each account (id,reply_count).")
    args = parser.parse_args()
    cresci15 = args.dataset == "cresci-15"
    if args.max_length is None:
        args.max_length = 128 if cresci15 else 512
    if args.batch_size < 1 or not 2 <= args.max_length <= 512:
        parser.error("--batch-size must be positive and --max-length must be between 2 and 512")
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.include_support and cresci15:
        parser.error("--include-support is available for TwiBot-20 only.")
    if args.include_support and args.reuse_embeddings:
        parser.error("--reuse-embeddings expects labeled-only input.")
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    reply_counts = read_reply_counts(args.reply_counts) if args.reply_counts else None
    if cresci15:
        users, split_sizes, edge_index, edge_type = read_cresci15(args.raw_dir)
        graph_rows = users
    else:
        users = RawAccounts(args.raw_dir, include_support=args.include_support)
        graph_rows, split_sizes = users.graph_rows()
        edge_index, edge_type = follow_graph(graph_rows)
    labels = torch.tensor([row["label"] for row in graph_rows], dtype=torch.long)
    n_users = len(graph_rows)
    del graph_rows
    snapshot = CRESCI_SNAPSHOT if cresci15 else SNAPSHOT
    numeric, categorical = profile_features(users, snapshot, log_counts=cresci15)
    print(f"[data] {n_users} accounts, {edge_type.numel()} directed edges", flush=True)
    print("[features] Computing lifecycle statistics", flush=True)
    lifecycle = extract_lifecycle_features(users, edge_index, n_users, snapshot, reply_counts=reply_counts)
    measured_replies = lifecycle["interactive"][:, -1].ge(0).sum().item()
    print(f"[features] Lifetime reply counts available for {measured_replies}/{n_users} accounts", flush=True)
    if args.reuse_embeddings:
        description, tweets = reuse_embeddings(args.reuse_embeddings, users, labels)
    else:
        description = encode_users(bios(users), n_users, args.description_model, args)
        tweets = encode_users(tweet_texts(users), n_users, args.tweet_model, args)

    save_dataset({
        "format_version": DATASET_VERSION,
        "dataset": args.dataset,
        "split_sizes": split_sizes,
        "embedding_source": "imported" if args.reuse_embeddings else "generated",
        "description": description,
        "tweets": tweets,
        "numeric": numeric,
        "categorical": categorical,
        "labels": labels,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "lifecycle": lifecycle,
    }, args.output)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
