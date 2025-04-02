#!/usr/bin/env python3
"""
Classification script for Spanish-English code-switching detection.

Supports:
  - Training multiple models (SGD + NB).
  - Predicting an unlabeled .conll test set.
  - Saving and loading models from disk.

Example training commands:
  python3 classify.py --mode train --data processed_data --output models --class-weights
  python classify.py --mode train --data processed_data --output test_models --sample 10000

Example prediction commands:
  python classify.py --mode predict --data processed_data --output models \
      --test-file lid_spaeng/test.conll --prediction-output predictions.conll
"""

import os
import sys
import time
import random
import argparse
import pickle
import joblib
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from preprocess import extract_features, load_dictionaries


label_encoder = LabelEncoder()


# This function loads the CSV features and the feature definition .pkl file


def load_preprocessed_data(
    data_path: str = 'processed_data',
    test_size: float = 0.2,
    sample_size: int = 0
):
    features_path = os.path.join(data_path, 'features.csv')
    feature_names_path = os.path.join(data_path, 'feature_names.pkl')

    # checks if files exist
    if not os.path.exists(features_path) or not os.path.exists(feature_names_path):
        print(f"Preprocessed data not found at {data_path}.")
        return None, None, None, None

    with open(feature_names_path, 'rb') as f:
        feature_data = pickle.load(f)
        if isinstance(feature_data, dict):
            features_list = feature_data.get('features', [])
            print(
                f"Loaded feature definitions ({len(features_list)} features).")
        else:
            print(
                f"Warning: Unexpected feature data type ({type(feature_data)}).")
            return None, None, None, None

    total_lines = 0
    with open(features_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    if total_lines <= 1:
        print(f"No valid data in {features_path}.")
        return None, None, None, None

    print(f"Found {total_lines-1} data rows.")

    use_subset = sample_size > 0 and sample_size < (total_lines-1)
    if use_subset:
        sample_percent = sample_size / (total_lines-1)
        print(f"Using a random subset of {sample_size} rows.")
    else:
        sample_percent = 1.0
        print(f"Using all {total_lines-1} rows.")

    X_data = []
    y_data = []
    label_counts = {}
    error_count = 0
    error_lines = []

    # reading lines from CSV
    with open(features_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for i, line in enumerate(tqdm(f, total=total_lines-1, unit="rows")):
            if use_subset and random.random() > sample_percent:
                continue
            try:
                if '"' in line:
                    import csv
                    row = next(csv.reader([line]))
                    if len(row) < 4:
                        error_count += 1
                        if len(error_lines) < 5:
                            error_lines.append(
                                f"Line {i+2}: fewer than 4 parts")
                        continue
                    sent_id, word, label = row[0], row[1], row[2]
                    rest = ",".join(row[3:])
                else:
                    parts = line.strip().split(',', 3)
                    if len(parts) < 4:
                        error_count += 1
                        if len(error_lines) < 5:
                            error_lines.append(
                                f"Line {i+2}: fewer than 4 parts")
                        continue
                    sent_id, word, label, rest = parts

                # convert features to ints
                feature_values = []
                for val in rest.split(','):
                    val = val.strip()
                    if not val:
                        continue
                    if (val.isdigit()) or (val.startswith('-') and val[1:].isdigit()):
                        feature_values.append(int(val))

                label_counts[label] = label_counts.get(label, 0) + 1
                X_data.append(feature_values)
                y_data.append(label)
            except Exception as e:
                error_count += 1
                if len(error_lines) < 5:
                    error_lines.append(f"Line {i+2} parsing error: {e}")

    print("\nLabel distribution in loaded data:")
    total_samples = len(y_data)
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        perc = (count / total_samples) * 100 if total_samples else 0
        print(f"  - {label}: {count} samples ({perc:.2f}%)")
    if error_count > 0:
        print(f"\nEncountered {error_count} parse errors. Examples:")
        for e in error_lines:
            print(f"  - {e}")

    # find the most common length of features and filter the rest
    length_counts = {}
    for v in X_data:
        length_counts[len(v)] = length_counts.get(len(v), 0) + 1
    most_common_length = max(length_counts.items(), key=lambda x: x[1])[0]
    valid_indices = [i for i, vec in enumerate(
        X_data) if len(vec) == most_common_length]
    X_data = [X_data[i] for i in valid_indices]
    y_data = [y_data[i] for i in valid_indices]

    X = np.array(X_data)
    y = np.array(y_data)
    print(f"Kept {len(X)} samples after standardization.\n")

    # final split into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(
        f"Final splits: {X_train.shape[0]} train, {X_val.shape[0]} validation.")
    print(f"Feature dimension: {X_train.shape[1]}")
    return X_train, X_val, y_train, y_val

# function to train models


def train_models(X_train: np.ndarray, y_train: np.ndarray, use_class_weights: bool = False):
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print("\nTraining set details:")
    print(f"  - {len(y_train)} samples total")
    print(f"  - {len(unique_classes)} unique classes")

    # we do class weights if needed
    if use_class_weights:
        cw = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight_dict = {cls: weight for cls,
                             weight in zip(unique_classes, cw)}
        for cls, wt in class_weight_dict.items():
            print(f"  - Class '{cls}': weight = {wt:.2f}")
    else:
        class_weight_dict = None

    # we define 3 models
    svm = SGDClassifier(
        loss='hinge',
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        alpha=0.0001,
        class_weight=class_weight_dict
    )
    nb = MultinomialNB(alpha=0.1)
    xgboost = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_classes),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        verbosity=1
    )

    models = {
        "LinearSVM": svm,
        "NaiveBayes": nb,
        "XGBoost": xgboost
    }

    # training them
    for name, model in models.items():
        if name == "XGBoost":
            y_train = label_encoder.fit_transform(y_train)

        print(f"\nTraining {name}...")
        start_t = time.time()
        model.fit(X_train, y_train)
        secs = time.time() - start_t
        print(f"  - {name} trained in {secs:.2f}s")
        if hasattr(model, 'classes_'):
            print(f"  - Classes: {model.classes_}")

    return models

# evaluate each model on validation


def evaluate_models(models: dict, X_val: np.ndarray, y_val: np.ndarray):
    print("\nEvaluating models on validation set...")
    all_models = dict(models)

    results = {}
    for name, model in all_models.items():
        if name == "XGBoost":
            global label_encoder
            y_val_encoded = label_encoder.transform(y_val)

            print(f"\nEvaluating {name}...")
            preds_encoded = model.predict(X_val)
            preds = label_encoder.inverse_transform(preds_encoded)

            acc = accuracy_score(y_val_encoded, preds_encoded)
            w_f1 = f1_score(y_val_encoded, preds_encoded, average='weighted')
            m_f1 = f1_score(y_val_encoded, preds_encoded, average='macro')

            results[name] = {'accuracy': acc, 'weighted_f1': w_f1, 'macro_f1': m_f1}
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - Weighted F1: {w_f1:.4f}")
            print(f"  - Macro F1: {m_f1:.4f}")
            print("  - Detailed classification report:")
            print(classification_report(y_val_encoded, preds_encoded, zero_division=0))

        else:
            print(f"\nEvaluating {name}...")
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            w_f1 = f1_score(y_val, preds, average='weighted')
            m_f1 = f1_score(y_val, preds, average='macro')
            results[name] = {'accuracy': acc,
                            'weighted_f1': w_f1, 'macro_f1': m_f1}
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - Weighted F1: {w_f1:.4f}")
            print(f"  - Macro F1: {m_f1:.4f}")
            print("  - Detailed classification report:")
            print(classification_report(y_val, preds, zero_division=0))

    best_model_name = max(
        results.items(), key=lambda x: x[1]['weighted_f1'])[0]
    print(
        f"\nBest model: {best_model_name} (weighted F1={results[best_model_name]['weighted_f1']:.4f})")
    return results, best_model_name

# save the models


def save_models(models: dict,  best_model: str,  output_path: str = 'models'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"\nSaving models to {output_path} (timestamp={timestamp})")
    for name, model in models.items():
        file_path = os.path.join(output_path, f"{name}_{timestamp}.joblib")
        print(f"  - Saving {name} => {file_path}")
        joblib.dump(model, file_path)

    # we only store best model as BestModel.joblib
    if best_model in models:
        best_model_obj = models[best_model]
    else:
        print(f"Warning: best_model '{best_model}' not recognized; skipping.")
        return
    best_path = os.path.join(output_path, f"BestModel.joblib")
    print(f"  - Saving Best Model => {best_path}")
    joblib.dump(best_model_obj, best_path)

# parse an unlabeled conll file


def parse_unlabeled_conll(file_path: str) -> dict:
    sentences = {}
    current_sent = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# sent_enum ='):
                current_sent = int(line.split('=')[1].strip())
                sentences[current_sent] = []
            else:
                if current_sent is None:
                    current_sent = 0 if len(
                        sentences) == 0 else max(sentences)+1
                    sentences[current_sent] = []
                sentences[current_sent].append(line)
    return sentences


def predict_conll_file(
    test_file: str,
    output_file: str,
    model_path: str,
    feature_path: str
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature definitions not found at {feature_path}")
    model = joblib.load(model_path)
    with open(feature_path, 'rb') as f:
        feature_data = pickle.load(f)
    english_words, spanish_words = load_dictionaries()
    print(f"\nPredicting labels for file: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line_stripped = line.strip()
            if not line_stripped:
                f_out.write("\n")
                continue
            if line_stripped.startswith('# sent_enum ='):
                f_out.write(line)
                current_sent = int(line_stripped.split('=')[1].strip())
                continue
            # we extract features for each line
            feats = extract_features(
                line_stripped,
                english_words,
                spanish_words,
                feature_data['ngrams'],
                {f: idx for idx, f in enumerate(feature_data['features'])}
            )
            feats = feats.reshape(1, -1)
            pred_label = model.predict(feats)[0]
            f_out.write(f"{line_stripped} {pred_label}\n")
    print(f"Predictions written to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description="Language Identification Pipeline")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'evaluate'],
                        help='Mode: train, predict, or evaluate.')
    parser.add_argument('--data', type=str, default='processed_data')
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--test-file', type=str, default=None)
    parser.add_argument('--prediction-output', type=str, default=None)
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    if args.mode == 'train':
        X_train, X_val, y_train, y_val = load_preprocessed_data(
            data_path=args.data, sample_size=args.sample
        )
        if X_train is None:
            print("No data loaded.")
            sys.exit(1)
        # train the models
        models = train_models(
            X_train, y_train, use_class_weights=args.class_weights)
        # evaluate them
        results, best_model = evaluate_models(models, X_val, y_val)
        # save them
        save_models(models, best_model,
                    output_path=args.output)

    elif args.mode == 'predict':
        if not args.test_file:
            print("Error: For batch prediction, specify --test-file test.conll")
            sys.exit(1)
        model_path = os.path.join(args.output, 'BestModel.joblib')
        if not os.path.exists(model_path):
            print("No BestModel.joblib found. Exiting.")
            sys.exit(1)
        pred_out = args.prediction_output
        if not pred_out:
            pred_out = os.path.join(args.output, "test_predictions.conll")
        # do the prediction
        predict_conll_file(
            test_file=args.test_file,
            output_file=pred_out,
            model_path=model_path,
            feature_path=os.path.join(args.data, 'feature_names.pkl')
        )

    print("\nProcess complete.")



if __name__ == "__main__":
    sys.exit(main())
