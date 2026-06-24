import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)

from pronunciation_assessment import (
    find_recordings, extract_features, compute_pronunciation_score,
    DATA_DIR, REFERENCE_SPEAKER
)

RANDOM_STATE = 42
N_FOLDS = 5  # reduce to 3 if any class has fewer than 5 samples total


# ============================================================
# 1. BUILD FEATURE TABLE (one row per recording, label = correct/incorrect)
# ============================================================
def build_feature_table(data_dir=DATA_DIR, reference_speaker=REFERENCE_SPEAKER):
    records = find_recordings(data_dir)
    ref_feats_by_word = {}
    rows = []

    # cache reference features per word (needed for the rule-based score)
    for word, label_map in records[reference_speaker].items():
        if "correct" in label_map:
            ref_feats_by_word[word] = extract_features(label_map["correct"])

    for speaker, word_map in records.items():
        for word, label_map in word_map.items():
            if word not in ref_feats_by_word:
                continue
            for label, path in label_map.items():
                if speaker == reference_speaker and label == "correct":
                    continue  # this IS the reference, skip self-comparison

                feats = extract_features(path)
                mfcc_mean = feats["mfcc_mean"]  # 12 values

                row = {
                    "speaker": speaker,
                    "word": word,
                    "label": label,  # ground truth: correct / incorrect
                }
                for i, v in enumerate(mfcc_mean):
                    row[f"mfcc_{i+1}"] = v
                row["pitch_mean"] = feats["pitch_mean"]
                row["duration"] = feats["duration"]
                row["F1"] = feats["formants"]["F1"]
                row["F2"] = feats["formants"]["F2"]
                row["F3"] = feats["formants"]["F3"]
                row["energy_mean"] = feats["energy_mean"]

                # rule-based score + sub-scores against this word's reference
                breakdown = compute_pronunciation_score(ref_feats_by_word[word], feats)
                row["rule_based_score"] = breakdown["final_score"]
                row["mfcc_similarity"] = breakdown["mfcc_similarity"]
                row["pitch_similarity"] = breakdown["pitch_similarity"]
                row["duration_similarity"] = breakdown["duration_similarity"]
                row["formant_similarity"] = breakdown["formant_similarity"]

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("feature_table.csv", index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 2. RULE-BASED CLASSIFIER (CV-tuned threshold, not a guessed constant)
# ============================================================
def rule_based_predict(scores, threshold):
    return np.where(scores >= threshold, "correct", "incorrect")


def best_threshold_for_fold(train_scores, train_labels, n_candidates=200):
    """Grid-search the threshold that maximizes accuracy on the TRAIN fold only.
    This keeps the rule-based method honest: the threshold is fit on training
    data and evaluated on a held-out fold, exactly like the ML models below."""
    lo, hi = train_scores.min(), train_scores.max()
    candidates = np.linspace(lo, hi, n_candidates)
    best_t, best_acc = candidates[0], -1
    for t in candidates:
        pred = rule_based_predict(train_scores, t)
        acc = accuracy_score(train_labels, pred)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


# ============================================================
# 3. UNIFIED CV EVALUATION (Rule-Based + ML share the SAME folds)
# ============================================================
# Set A: raw acoustic features - absolute values, carry speaker identity (timbre, pitch range)
RAW_FEATURE_COLS = [f"mfcc_{i+1}" for i in range(12)] + \
                    ["pitch_mean", "duration", "F1", "F2", "F3", "energy_mean"]

# Set B: relative similarity-to-reference scores - normalized against the reference speaker,
# so absolute speaker timbre/pitch range is mostly removed; closer to "pronunciation accuracy"
SIMILARITY_FEATURE_COLS = ["mfcc_similarity", "pitch_similarity",
                           "duration_similarity", "formant_similarity"]


def make_cv_splits(y, n_folds):
    min_class_count = min(np.bincount(y))
    folds = max(min(n_folds, min_class_count), 2)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    # splitting only needs len(y) and the stratification labels - independent of which
    # feature set we later evaluate, so the SAME splits are reused for everything below
    return list(skf.split(np.zeros(len(y)), y)), folds


def evaluate_rule_based_cv(df, le, y, split_indices):
    rb_scores = df["rule_based_score"].values
    y_pred = np.empty_like(y)
    used_thresholds = []

    for train_idx, test_idx in split_indices:
        t = best_threshold_for_fold(rb_scores[train_idx], le.inverse_transform(y[train_idx]))
        used_thresholds.append(t)
        pred = rule_based_predict(rb_scores[test_idx], t)
        y_pred[test_idx] = le.transform(pred)

    print(f"Rule-based thresholds chosen per fold: {[round(t, 1) for t in used_thresholds]}")
    return {"y_true": le.inverse_transform(y), "y_pred": le.inverse_transform(y_pred)}


def evaluate_ml_cv(df, feature_cols, le, y, split_indices):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
    }
    y_pred = {name: np.empty_like(y) for name in models}

    for train_idx, test_idx in split_indices:
        for name, model in models.items():
            model.fit(X_scaled[train_idx], y[train_idx])
            y_pred[name][test_idx] = model.predict(X_scaled[test_idx])

    return {name: {"y_true": le.inverse_transform(y), "y_pred": le.inverse_transform(y_pred[name])}
            for name in models}


# ============================================================
# 4. COMPARISON TABLE + PLOTS
# ============================================================
def summarize(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="correct")
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "method": name,
        "accuracy": round(acc, 3),
        "f1_correct": round(f1, 3),
        "precision_correct": round(report["correct"]["precision"], 3),
        "recall_correct": round(report["correct"]["recall"], 3),
    }


def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=["correct", "incorrect"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["correct", "incorrect"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["correct", "incorrect"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"plots/confusion_{safe_name}.png", dpi=120)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Building feature table from data/ ...")
    df = build_feature_table()
    print(f"Total samples: {len(df)}  |  correct: {(df['label']=='correct').sum()}  "
          f"incorrect: {(df['label']=='incorrect').sum()}")

    summary_rows = []

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    split_indices, folds_used = make_cv_splits(y, N_FOLDS)
    print(f"\nUsing {folds_used}-fold Stratified Cross-Validation (identical splits for all methods below).")

    # --- Rule-Based (CV-tuned threshold) - evaluated once, doesn't depend on feature set ---
    rb_res = evaluate_rule_based_cv(df, le, y, split_indices)
    summary_rows.append(summarize("Rule-Based (CV-tuned threshold)", rb_res["y_true"], rb_res["y_pred"]))
    plot_confusion(rb_res["y_true"], rb_res["y_pred"], "Rule-Based")

    # --- Set A: raw acoustic features (absolute - may leak speaker identity) ---
    print("\n>>> Feature Set A: RAW acoustic features (mfcc, pitch, formants - absolute)")
    results_raw = evaluate_ml_cv(df, RAW_FEATURE_COLS, le, y, split_indices)
    for name, res in results_raw.items():
        tag = f"{name} [raw features]"
        summary_rows.append(summarize(tag, res["y_true"], res["y_pred"]))
        plot_confusion(res["y_true"], res["y_pred"], tag)

    # --- Set B: similarity-to-reference features (relative - less speaker leakage) ---
    print("\n>>> Feature Set B: SIMILARITY-TO-REFERENCE features (relative)")
    results_sim = evaluate_ml_cv(df, SIMILARITY_FEATURE_COLS, le, y, split_indices)
    for name, res in results_sim.items():
        tag = f"{name} [similarity features]"
        summary_rows.append(summarize(tag, res["y_true"], res["y_pred"]))
        plot_confusion(res["y_true"], res["y_pred"], tag)

    comparison_df = pd.DataFrame(summary_rows)
    comparison_df.to_csv("method_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n=== Rule-Based vs ML Comparison ===")
    print(comparison_df.to_string(index=False))
    print("\nSaved: feature_table.csv, method_comparison.csv, plots/confusion_*.png")