#!/usr/bin/env python3
"""
Classification script for Spanish-English code-switching detection.

Supports:
  - Training multiple models (SGD + NB + XGBoost + Stacking). # Added Stacking
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
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from preprocess import extract_features, load_dictionaries


def load_preprocessed_data(
    data_path: str = 'processed_data',
    test_size: float = 0.2,
    sample_size: int = 0
):
    """"
    Load preprocessed data from CSV and feature definitions from a pickle file."
    """
    features_path = os.path.join(data_path, 'features.csv')
    feature_names_path = os.path.join(data_path, 'feature_names.pkl')

    # checks if files exist
    if not os.path.exists(features_path) or not os.path.exists(feature_names_path):
        print(f"Preprocessed data not found at {data_path}.")
        return None, None, None, None, None

    with open(feature_names_path, 'rb') as f:
        feature_data = pickle.load(f)
        if isinstance(feature_data, dict):
            features_list = feature_data.get('features', [])
            print(
                f"Loaded feature definitions ({len(features_list)} features).")
        else:
            print(
                f"Warning: Unexpected feature data type ({type(feature_data)}).")
            return None, None, None, None, None

    total_lines = 0
    with open(features_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    if total_lines <= 1:
        print(f"No valid data in {features_path}.")
        return None, None, None, None, None

    print(f"Found {total_lines-1} data rows.")

    # check subset usage
    use_subset = sample_size > 0 and sample_size < (total_lines-1)
    if use_subset:
        skip_probability = 1.0 - (sample_size / (total_lines - 1))
        print(f"Using a random subset of approx {sample_size} rows.")
    else:
        skip_probability = 0.0
        print(f"Using all {total_lines-1} rows.")

    X_data = []
    y_data = []
    label_counts = {}
    error_count = 0
    error_lines = []

    # reading lines from CSV
    with open(features_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        effective_total = sample_size if use_subset else total_lines - 1
        pbar = tqdm(f, total=total_lines-1, unit="rows", desc="Loading data")
        line_num = 1 
        loaded_count = 0
        for line in pbar:
            line_num += 1
            # randomly skip lines based on probability if sampling
            if use_subset and random.random() < skip_probability:
                continue
            if use_subset and loaded_count >= sample_size:
                 continue # stop if enough samples

            try:
                if '"' in line:
                    import csv
                    try:
                         row = next(csv.reader([line]))
                    except StopIteration:
                         error_count +=1
                         if len(error_lines) < 5:
                             error_lines.append(f"Line {line_num}: Empty or unreadable")
                         continue
                    if len(row) < 4:
                        error_count += 1
                        if len(error_lines) < 5:
                            error_lines.append(
                                f"Line {line_num}: fewer than 4 parts (quoted)")
                        continue
                    sent_id, word, label = row[0], row[1], row[2]
                    rest = ",".join(row[3:]) 
                else:
                    parts = line.strip().split(',', 3)
                    if len(parts) < 4:
                        error_count += 1
                        if len(error_lines) < 5:
                            error_lines.append(
                                f"Line {line_num}: fewer than 4 parts (unquoted)")
                        continue
                    sent_id, word, label, rest = parts

                # convert features to ints
                feature_values = []
                feature_strings = rest.split(',')
                valid_row = True
                for val_str in feature_strings:
                    val_str = val_str.strip()
                    if not val_str: # skip empty feature strings
                       continue
                    try:
                        feature_values.append(int(val_str))
                        # fallback to if conversion fails
                    except ValueError:
                        error_count += 1
                        if len(error_lines) < 5:
                           error_lines.append(f"Line {line_num}: Non-integer feature '{val_str}'")
                        valid_row = False
                        break 
                if not valid_row:
                    continue # skip to next line

                label_counts[label] = label_counts.get(label, 0) + 1
                X_data.append(feature_values)
                y_data.append(label)
                loaded_count += 1
                pbar.set_postfix({"Loaded": loaded_count})

            except Exception as e:
                error_count += 1
                if len(error_lines) < 5:
                    error_lines.append(f"Line {line_num} parsing error: {e} | Content: {line.strip()}")
        pbar.close()

    print(f"\nSuccessfully loaded {len(y_data)} samples.")

    print("\nLabel distribution in loaded data:")
    total_samples = len(y_data)
    if total_samples > 0:
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            perc = (count / total_samples) * 100
            print(f"  - {label}: {count} samples ({perc:.2f}%)")
    else:
        print("No samples loaded.")
        return None, None, None, None, None

    if error_count > 0:
        print(f"\nEncountered {error_count} parse errors processing input file. Examples:")
        for e in error_lines:
            print(f"  - {e}")

    # find the most common length of features and filter the rest
    if not X_data: # check if empty
        print("No feature data loaded, cannot proceed.")
        return None, None, None, None, None

    length_counts = {}
    for v in X_data:
        length_counts[len(v)] = length_counts.get(len(v), 0) + 1

    if not length_counts:
         print("Feature length calculation failed (no data?).")
         return None, None, None, None, None

    # find the most common length
    most_common_length = max(length_counts.items(), key=lambda x: x[1])[0]
    print(f"Most common feature vector length: {most_common_length}")
    original_count = len(X_data)
    valid_indices = [i for i, vec in enumerate(X_data) if len(vec) == most_common_length]

    # check for valid indices
    if not valid_indices:
        print(f"Error: No samples found with the most common feature length ({most_common_length}). Check feature extraction.")
        print("Feature length counts:", length_counts)
        return None, None, None, None, None

    X_data_filtered = [X_data[i] for i in valid_indices]
    y_data_filtered = [y_data[i] for i in valid_indices]

    num_removed = original_count - len(X_data_filtered)
    if num_removed > 0:
        print(f"Removed {num_removed} samples with feature lengths != {most_common_length}.")

    X = np.array(X_data_filtered)
    y = np.array(y_data_filtered)
    print(f"Kept {len(X)} samples after length standardization.\n")

    if X.shape[0] == 0:
        print("No data remaining after filtering. Exiting.")
        return None, None, None, None, None

    # initialize the label encoder (needed for XGBoost and Stacking, bc these need numeric labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Labels encoded: {label_encoder.classes_}")

    # final split into train and val
    X_train, X_val, y_train, y_val, y_train_encoded, y_val_encoded = train_test_split(
        X, y, y_encoded, test_size=test_size, random_state=42, stratify=y.ravel()
    )

    print(
        f"Final splits: {X_train.shape[0]} train, {X_val.shape[0]} validation.")
    print(f"Feature dimension: {X_train.shape[1]}")
    return X_train, X_val, y_train, y_val, y_train_encoded, y_val_encoded, label_encoder


def train_base_models(X_train: np.ndarray, y_train: np.ndarray, y_train_encoded: np.ndarray, label_encoder: LabelEncoder, use_class_weights: bool = False):
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    '''
        train base models
    '''

    print("\n--- Training Base Models ---")
    print(f"Training set details:")
    print(f"  - {len(y_train)} samples total")
    print(f"  - {len(unique_classes)} unique classes: {unique_classes}")

    # we do class weights if needed
    class_weight_dict = None
    if use_class_weights:
        cw = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight_dict = {cls: weight for cls,
                             weight in zip(unique_classes, cw)}
        print("Using balanced class weights:")
        for cls, wt in class_weight_dict.items():
            print(f"  - Class '{cls}': weight = {wt:.2f}")
    else:
        print("Not using class weights.")
        class_weight_dict = None

    # we define 3 base models: svm, multinomialNB and XGBoost
    svm = SGDClassifier(
        loss='hinge',
        max_iter=5000, 
        tol=1e-4, 
        random_state=42,
        n_jobs=1,
        verbose=0,  
        alpha=0.0001,
        class_weight=class_weight_dict
    )

    nb = MultinomialNB(alpha=0.1)

    xgboost = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_classes),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=150,
        use_label_encoder=False, 
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1,
        verbosity=0
    )

    models = {
        "LinearSVM": svm,
        "NaiveBayes": nb,
        "XGBoost": xgboost
    }

    # training loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_t = time.time()
        if name == "XGBoost":
            # XGBoost needs encoded labels
            model.fit(X_train, y_train_encoded.ravel())
        else:
            # SVM and NB use original labels
            model.fit(X_train, y_train.ravel())
        secs = time.time() - start_t
        print(f"  - {name} trained in {secs:.2f}s")
        if hasattr(model, 'classes_'):
            print(f"  - Learned classes: {model.classes_}")

    return models

# evaluate each model on validation
def evaluate_models(models: dict, X_train: np.ndarray, y_train_encoded: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, y_val_encoded: np.ndarray, label_encoder: LabelEncoder):
    print("\n--- Evaluating Models ---")
    all_models_results = {}

    # --- Evaluate Base Models ---
    print("\nEvaluating Base Models on Validation Set...")
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        start_t = time.time()
        preds = model.predict(X_val)
        pred_time = time.time() - start_t

        y_true_eval = y_val
        preds_eval = preds

        # for XGBoost, we use the encoded labels for evaluation again
        if name == "XGBoost":
            preds_eval = label_encoder.inverse_transform(preds)

        acc = accuracy_score(y_true_eval, preds_eval)
        w_f1 = f1_score(y_true_eval, preds_eval, average='weighted', zero_division=0)
        m_f1 = f1_score(y_true_eval, preds_eval, average='macro', zero_division=0)

        all_models_results[name] = {'accuracy': acc, 'weighted_f1': w_f1, 'macro_f1': m_f1}

        print(f"  - Prediction time: {pred_time:.2f}s")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - Weighted F1: {w_f1:.4f}")
        print(f"  - Macro F1: {m_f1:.4f}")
        print("  - Classification Report (on original labels):")
        print(classification_report(y_true_eval, preds_eval, target_names=label_encoder.classes_, zero_division=0))

    # --- Train and Evaluate Stacking Classifier ---
    print("\n--- Training and Evaluating Stacking Classifier ---")

    estimators = [
        ('LinearSVM', SGDClassifier(loss='hinge', max_iter=5000, tol=1e-4, random_state=42, n_jobs=1, verbose=0, alpha=0.0001)),
        ('NaiveBayes', MultinomialNB(alpha=0.1)),
        ('XGBoost', xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), max_depth=6, learning_rate=0.1, n_estimators=150, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=1, verbosity=0))
    ]

    # define meta-learner
    meta_learner = CatBoostClassifier(
        iterations=150,
        learning_rate=0.05, 
        depth=6,
        verbose=0,
        random_state=42,
        loss_function='MultiClass'
    )
    

    # Define Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=1, 
        passthrough=True 
    )

    print("Training Stacking Classifier (using internal CV)...")
    start_t = time.time()
    stacking_model.fit(X_train, y_train_encoded)
    train_time = time.time() - start_t
    print(f"  - Stacking Classifier trained in {train_time:.2f}s")

    print("\nEvaluating Stacking Classifier on Validation Set...")
    start_t = time.time()
    stacking_preds_encoded = stacking_model.predict(X_val)
    pred_time = time.time() - start_t

    y_true_encoded_eval = y_val_encoded
    preds_encoded_eval = stacking_preds_encoded

    acc = accuracy_score(y_true_encoded_eval, preds_encoded_eval)
    w_f1 = f1_score(y_true_encoded_eval, preds_encoded_eval, average='weighted', zero_division=0)
    m_f1 = f1_score(y_true_encoded_eval, preds_encoded_eval, average='macro', zero_division=0)

    stacking_model_name = "StackingClassifier"
    all_models_results[stacking_model_name] = {'accuracy': acc, 'weighted_f1': w_f1, 'macro_f1': m_f1}

    print(f"  - Prediction time: {pred_time:.2f}s")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Weighted F1: {w_f1:.4f}")
    print(f"  - Macro F1: {m_f1:.4f}")
    print("  - Classification Report (on original labels):")
    stacking_preds_orig = label_encoder.inverse_transform(stacking_preds_encoded)
    print(classification_report(y_val, stacking_preds_orig, target_names=label_encoder.classes_, zero_division=0))

    # --- Determine Best Model ---
    models[stacking_model_name] = stacking_model 

    # find best model based on weighted F1
    best_model_name = max(all_models_results.items(), key=lambda item: item[1]['weighted_f1'])[0]
    print(f"\n--- Best Model ---")
    print(f"Best performing model on validation set: {best_model_name} (Weighted F1 = {all_models_results[best_model_name]['weighted_f1']:.4f})")

    return all_models_results, best_model_name


# save the models
def save_models(models: dict,  best_model_name: str, label_encoder: LabelEncoder, output_path: str = 'models'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"\nSaving models and label encoder to {output_path} (timestamp={timestamp})")

    for name, model in models.items():
        file_path = os.path.join(output_path, f"{name}_{timestamp}.joblib")
        print(f"  - Saving {name} => {file_path}")
        try:
             joblib.dump(model, file_path)
        except Exception as e:
             print(f"    ERROR saving {name}: {e}")


    # we only store the best model as BestModel.joblib
    if best_model_name in models:
        best_model_obj = models[best_model_name]
        best_path = os.path.join(output_path, f"BestModel.joblib") 
        best_path_ts = os.path.join(output_path, f"BestModel_{timestamp}.joblib") 
        print(f"  - Saving Best Model ({best_model_name}) => {best_path} (and timestamped version)")
        try:
            joblib.dump(best_model_obj, best_path)
            joblib.dump(best_model_obj, best_path_ts) 
        except Exception as e:
            print(f"    ERROR saving Best Model: {e}")
    else:
        print(f"Warning: best_model_name '{best_model_name}' not found in models dictionary; skipping saving BestModel.joblib.")

    encoder_path = os.path.join(output_path, f"LabelEncoder_{timestamp}.joblib")
    print(f"  - Saving LabelEncoder => {encoder_path}")
    try:
        joblib.dump(label_encoder, encoder_path)
        joblib.dump(label_encoder, os.path.join(output_path, "LabelEncoder.joblib"))
    except Exception as e:
        print(f"    ERROR saving LabelEncoder: {e}")


# parse an unlabeled conll file
def parse_unlabeled_conll(file_path: str) -> dict:
    sentences = {}
    current_sent = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                current_sent = None
                continue
            if line.startswith('# sent_enum ='):
                try:
                    current_sent = int(line.split('=')[1].strip())
                    if current_sent not in sentences:
                         sentences[current_sent] = []
                except (IndexError, ValueError):
                     print(f"Warning: Could not parse sent_enum on line {line_num}: {line}")
                     current_sent = None
            elif line.startswith('#'):
                 continue
            else:
                if current_sent is None:
                     if not sentences: 
                         current_sent = 1 
                     else:
                         current_sent = max(sentences.keys()) + 1
                     print(f"Warning: Token on line {line_num} ('{line.split()[0]}...') found without preceding '# sent_enum ='. Assigning to sentence {current_sent}.")
                     sentences[current_sent] = []
                sentences[current_sent].append(line)
    return sentences


def predict_conll_file(
    test_file: str,
    output_file: str,
    model_path: str,
    label_encoder_path: str,
    feature_path: str
):
    """"
    Predict labels for a .conll file using a trained model and save predictions to a new file."
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"LabelEncoder not found at {label_encoder_path}")
    print(f"Loading label encoder from: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature definitions (feature_names.pkl) not found at {feature_path}")
    print(f"Loading feature definitions from: {feature_path}")
    with open(feature_path, 'rb') as f:
        feature_data = pickle.load(f)
        if isinstance(feature_data, dict) and 'features' in feature_data:
            feature_mapping = {f: idx for idx, f in enumerate(feature_data['features'])}
            print(f"Loaded {len(feature_mapping)} feature definitions.")
        else:
             raise ValueError(f"Invalid format in feature definitions file: {feature_path}")

    print("Loading language dictionaries...")
    try:
        english_words, spanish_words = load_dictionaries()
    except Exception as e:
        print(f"Warning: Could not load dictionaries. Feature extraction might be affected. Error: {e}")
        english_words, spanish_words = set(), set()


    print(f"\nPredicting labels for file: {test_file}")
    print(f"Writing predictions to: {output_file}")

    expected_feature_length = len(feature_mapping)
    print(f"Expecting {expected_feature_length} features per token.")

    lines_processed = 0
    errors_extracting = 0
    output_lines = [] 

    with open(test_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(tqdm(f_in, desc="Predicting", unit="lines")):
            line_stripped = line.strip()

            if not line_stripped:
                output_lines.append("")
                continue

            if line_stripped.startswith('#'):
                output_lines.append(line_stripped)
                continue

            lines_processed += 1
            # we extract features for each line
            try:
                feats_vector = extract_features(
                    line_stripped,
                    english_words,
                    spanish_words,
                    feature_data.get('ngrams', {}),
                    feature_mapping
                )

                if feats_vector is None:
                     print(f"Warning: Feature extraction returned None for line {line_num+1}: {line_stripped}")
                     errors_extracting += 1
                     predicted_label = "UNK_PRED"
                elif len(feats_vector) != expected_feature_length:
                     print(f"Warning: Feature length mismatch for line {line_num+1}. Expected {expected_feature_length}, got {len(feats_vector)}. Line: {line_stripped}")
                     errors_extracting += 1
                     predicted_label = "UNK_PRED_LEN" 
                else:
                     feats_reshaped = feats_vector.reshape(1, -1)
                     pred_encoded = model.predict(feats_reshaped)[0]
                     predicted_label = label_encoder.inverse_transform([pred_encoded])[0]

            except Exception as e:
                print(f"Error processing line {line_num+1}: {line_stripped}\n  Error: {e}")
                errors_extracting += 1
                predicted_label = "UNK_ERR"

            output_lines.append(f"{line_stripped}\t{predicted_label}")

    # write lines to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for out_line in output_lines:
             f_out.write(out_line + "\n")

    print(f"\nPrediction complete.")
    print(f"  - Processed {lines_processed} token lines.")
    if errors_extracting > 0:
        print(f"  - Encountered {errors_extracting} errors during feature extraction or prediction.")
    print(f"Predictions written to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description="Language Identification Pipeline")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'evaluate'],
                        help='Mode: train or predict.')
    parser.add_argument('--data', type=str, default='processed_data', help='Path to processed data directory (contains features.csv, feature_names.pkl)')
    parser.add_argument('--output', type=str, default='models', help='Path to save trained models and encoder.')
    parser.add_argument('--sample', type=int, default=0, help='Use a random subset of N samples for training (0=use all).')
    parser.add_argument('--class-weights', action='store_true', help='Use balanced class weights for training (affects SVM mainly).')
    parser.add_argument('--test-file', type=str, default=None, help='Path to the unlabeled .conll file for prediction.')
    parser.add_argument('--prediction-output', type=str, default=None, help='File path to save predictions.')
    parser.add_argument('--model-file', type=str, default=None, help='Specific model file to load (e.g., for predict mode, overrides BestModel.joblib).')
    parser.add_argument('--encoder-file', type=str, default=None, help='Specific LabelEncoder file to load (e.g., for predict mode, overrides LabelEncoder.joblib).')


    args = parser.parse_args()

    print(f"--- Running Mode: {args.mode} ---")

    # training mode
    if args.mode == 'train':
        print("\n--- Data Loading ---")
        load_result = load_preprocessed_data(
            data_path=args.data, sample_size=args.sample
        )

        if load_result is None or load_result[0] is None:
            print("Data loading failed. Exiting.")
            sys.exit(1)

        X_train, X_val, y_train, y_val, y_train_encoded, y_val_encoded, label_encoder = load_result

        # train base models
        base_models = train_base_models(X_train, y_train, y_train_encoded, label_encoder, args.class_weights)

        # evaluate them
        results, best_model_name = evaluate_models(
            base_models, X_train, y_train_encoded, X_val, y_val, y_val_encoded, label_encoder
            )
        all_trained_models = base_models

        # save models
        save_models(all_trained_models, best_model_name, label_encoder, output_path=args.output)

    # prediction mode
    elif args.mode == 'predict':
        print("\n--- Prediction ---")
        if not args.test_file:
            print("Error: For prediction mode, specify the input file using --test-file <path/to/your/test.conll>")
            sys.exit(1)
        if not os.path.exists(args.test_file):
             print(f"Error: Test file not found: {args.test_file}")
             sys.exit(1)

        model_path = args.model_file if args.model_file else os.path.join(args.output, 'BestModel.joblib')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at the specified or default location: {model_path}")
            print("Ensure you have trained a model first or provided the correct path using --model-file.")
            sys.exit(1)

        encoder_path = args.encoder_file if args.encoder_file else os.path.join(args.output, 'LabelEncoder.joblib')
        if not os.path.exists(encoder_path):
            print(f"Error: LabelEncoder file not found at the specified or default location: {encoder_path}")
            print("Ensure you have trained a model first (which saves the encoder) or provided the correct path using --encoder-file.")
            sys.exit(1)

        feature_path = os.path.join(args.data, 'feature_names.pkl')
        if not os.path.exists(feature_path):
             print(f"Error: Feature definitions file not found: {feature_path}")
             sys.exit(1)

        pred_out = args.prediction_output
        output_dir = os.path.dirname(pred_out)

        if output_dir:  #
            os.makedirs(output_dir, exist_ok=True)
            print(f"Ensured output directory exists: {output_dir}")
        else:
            print(f"Output file '{pred_out}' will be saved in the current directory.")

        # do the prediction
        predict_conll_file(
            test_file=args.test_file,
            output_file=pred_out,
            model_path=model_path,
            label_encoder_path=encoder_path,
            feature_path=feature_path
        )

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    main()
