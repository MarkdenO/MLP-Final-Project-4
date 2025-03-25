#!/usr/bin/env python3
"""
Preprocessing script for Spanish-English code-switching detection.
Parses a CoNLL file, extracts features, and writes them to a CSV file,
along with a pickle file containing feature definitions.

Usage:
    python3 preprocess.py --data lid_spaeng --output processed_data
"""

import os
import re
import argparse
import pickle
import string
import gc
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

# parse_conll_file reads a .conll file with format:
# # sent_enum = X
# token  label
# ...
# and returns a dict of sentence_id -> list of (word, label)


def parse_conll_file(file_path: str) -> dict:
    sentences = defaultdict(list)
    current_sent = None
    total_tokens = 0

    print(f"Parsing CoNLL file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# sent_enum ="):
                # each new sentence id
                current_sent = int(line.split("=")[1].strip())
            else:
                parts = re.split(r'\s+', line, maxsplit=1)
                if len(parts) == 2:
                    word, label = parts
                    sentences[current_sent].append((word, label))
                    total_tokens += 1
                else:
                    print(f"Warning: Wrong line format: '{line}'")
    print(
        f"Finished parsing: {total_tokens} tokens in {len(sentences)} sentences.")
    return dict(sentences)

# load word lists for English and Spanish, if available


def load_dictionaries(
    english_dict_path='languages/en.txt',
    spanish_dict_path='languages/es.txt'
) -> tuple[set, set]:
    if os.path.exists(english_dict_path) and os.path.exists(spanish_dict_path):
        with open(english_dict_path, 'r', encoding='utf-8') as f:
            english_words = set(line.strip().lower() for line in f)
        with open(spanish_dict_path, 'r', encoding='utf-8') as f:
            spanish_words = set(line.strip().lower() for line in f)
        return english_words, spanish_words
    print("Dictionary files not found. Returning empty sets.")
    return set(), set()

# detect if a string has any emoji characters


def detect_emoji(text: str) -> bool:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(text))

# check if text is all punctuation (like "...")


def is_punctuation(text: str) -> bool:
    return all(c in string.punctuation for c in text)

# checks if text is numeric


def is_numeric(text: str) -> bool:
    return bool(re.match(r'^[\d\.:]+$', text))

# checks for Spanish diacritics etc.


def has_spanish_characters(text: str) -> bool:
    spanish_chars = set('áéíóúüñ¿¡')
    return any(char in spanish_chars for char in text.lower())

# collects frequent n-grams up to length 3 in entire dataset


def collect_common_ngrams(sentences: dict, max_ngrams=500) -> dict:
    ngram_counters = {1: Counter(), 2: Counter(), 3: Counter()}
    print("Collecting most common n-grams...")
    for _, tokens in tqdm(sentences.items(), desc="N-gram collection"):
        for word, _ in tokens:
            word_lower = word.lower()
            for n in range(1, 4):
                for i in range(len(word_lower) - n + 1):
                    ngram = word_lower[i:i+n]
                    ngram_counters[n][ngram] += 1
    common_ngrams_map = {}
    for n in range(1, 4):
        most_common = ngram_counters[n].most_common(max_ngrams)
        for ng, _freq in most_common:
            common_ngrams_map[ng] = 1
    print(f"Selected {len(common_ngrams_map)} total unique n-grams.")
    return common_ngrams_map

# This function extracts features for a single token


def extract_features(
    word: str,
    english_words: set,
    spanish_words: set,
    common_ngrams: dict,
    feature_index: dict
) -> np.ndarray:
    vec = np.zeros(len(feature_index), dtype=np.int8)
    word_lower = word.lower()
    if 'word_length' in feature_index:
        vec[feature_index['word_length']] = min(len(word), 255)
    if 'starts_with_hashtag' in feature_index:
        vec[feature_index['starts_with_hashtag']
            ] = 1 if word.startswith('#') else 0
    if 'starts_with_at' in feature_index:
        vec[feature_index['starts_with_at']] = 1 if word.startswith('@') else 0
    if 'is_url' in feature_index:
        vec[feature_index['is_url']] = 1 if word.startswith(
            'http') or 'www' in word else 0
    if 'is_emoji' in feature_index:
        if detect_emoji(word):
            vec[feature_index['is_emoji']] = 1
    if 'is_punctuation' in feature_index:
        if is_punctuation(word):
            vec[feature_index['is_punctuation']] = 1
    if 'is_numeric' in feature_index:
        if is_numeric(word):
            vec[feature_index['is_numeric']] = 1
    if 'has_spanish_chars' in feature_index:
        if has_spanish_characters(word):
            vec[feature_index['has_spanish_chars']] = 1
    if 'in_english_dict' in feature_index:
        if word_lower in english_words:
            vec[feature_index['in_english_dict']] = 1
    if 'in_spanish_dict' in feature_index:
        if word_lower in spanish_words:
            vec[feature_index['in_spanish_dict']] = 1
    if 'possible_named_entity' in feature_index:
        if len(word) > 1 and word[0].isupper() and not word.isupper():
            vec[feature_index['possible_named_entity']] = 1
    for n in range(1, 4):
        for i in range(len(word_lower) - n + 1):
            ngram = word_lower[i:i+n]
            if ngram in common_ngrams:
                idx_ngram = feature_index.get(f'contains_{ngram}')
                if idx_ngram is not None:
                    vec[idx_ngram] = 1
    return vec

# This is the main function for building the CSV from the train.conll data


def preprocess_data(
    data_path: str,
    output_path: str = 'processed_data',
    batch_size: int = 5000
) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    english_words, spanish_words = load_dictionaries()
    conll_file = os.path.join(data_path, "train.conll")
    if not os.path.isfile(conll_file):
        raise FileNotFoundError(
            f"Could not find training file at: {conll_file}")
    sentences = parse_conll_file(conll_file)
    common_ngrams = collect_common_ngrams(sentences, max_ngrams=300)
    base_features = [
        'word_length', 'starts_with_hashtag', 'starts_with_at', 'is_url',
        'is_emoji', 'is_punctuation', 'is_numeric', 'has_spanish_chars',
        'in_english_dict', 'in_spanish_dict', 'possible_named_entity'
    ]
    ngram_features = [f'contains_{ng}' for ng in common_ngrams]
    all_features = base_features + ngram_features
    feature_index = {feat: i for i, feat in enumerate(all_features)}
    feature_names = {
        'features': all_features,
        'ngrams': common_ngrams
    }
    feature_names_path = os.path.join(output_path, 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    all_tokens = [(sid, w, lbl) for sid, tokens in sentences.items()
                  for w, lbl in tokens]
    total_tokens = len(all_tokens)
    total_batches = (total_tokens + batch_size - 1) // batch_size
    print(
        f"Processing {total_tokens} tokens in {total_batches} batches of size {batch_size}.")
    output_csv = os.path.join(output_path, 'features.csv')
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("sentence_id,word,label,features\n")

    # batch processing to handle large data
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_tokens)
        print(
            f"  - Batch {batch_idx+1}/{total_batches} (rows {start_idx+1} to {end_idx})")
        batch_tokens = all_tokens[start_idx:end_idx]
        batch_data = []
        for sent_id, word, label in tqdm(batch_tokens, desc=f"Batch {batch_idx+1}"):
            # extract features for each token using the same logic
            feature_vector = extract_features(
                word, english_words, spanish_words, common_ngrams, feature_index)
            feature_str = ",".join(str(v) for v in feature_vector)
            label_safe = f'"{label}"' if ',' in label else label
            word_safe = f'"{word}"' if ',' in word else word
            batch_data.append((sent_id, word_safe, label_safe, feature_str))

        with open(output_csv, 'a', encoding='utf-8') as f:
            for row in batch_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
        del batch_data
        gc.collect()

    print(f"\nPreprocessing complete! {total_tokens} tokens processed.")
    print(f"Output CSV: {output_csv}")
    print(f"Feature names: {feature_names_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='processed_data')
    parser.add_argument('--batch_size', type=int, default=10000)
    args = parser.parse_args()
    preprocess_data(args.data, args.output, args.batch_size)


if __name__ == "__main__":
    main()
