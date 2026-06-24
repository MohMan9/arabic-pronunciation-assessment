import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from dtw import dtw
import parselmouth
from parselmouth.praat import call

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "data"                 # root folder containing speaker subfolders
REFERENCE_SPEAKER = "speaker1"     # whose "correct" recordings are the reference/ground truth
PLOTS_DIR = "plots"
SR = 16000                         # target sample rate
N_MFCC = 12

WEIGHTS = {"mfcc": 0.40, "pitch": 0.20, "duration": 0.20, "formant": 0.20}

os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# 1. PREPROCESSING
# ============================================================
def load_and_preprocess(path, sr=SR, top_db=30):
    """Load -> resample -> trim silence -> normalize amplitude."""
    signal, _ = librosa.load(path, sr=sr, mono=True)
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    if np.max(np.abs(trimmed)) > 0:
        trimmed = trimmed / np.max(np.abs(trimmed))
    return trimmed, sr


def frame_signal(signal, sr, frame_ms=25, hop_ms=10):
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    return frame_length, hop_length


# ============================================================
# 2. SHORT-TIME ANALYSIS (deliverable: plots)
# ============================================================
def short_time_energy(signal, frame_length, hop_length):
    energy = np.array([
        np.sum(np.abs(signal[i:i + frame_length] ** 2))
        for i in range(0, len(signal) - frame_length, hop_length)
    ])
    return energy


def zero_crossing_rate(signal, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(
        signal, frame_length=frame_length, hop_length=hop_length
    )[0]


def plot_short_time_analysis(signal, sr, frame_length, hop_length, tag):
    """Saves waveform + spectrogram + STE + ZCC plot. tag = identifying filename."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    librosa.display.waveshow(signal, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform - {tag}")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    ste = short_time_energy(signal, frame_length, hop_length)
    axes[2].plot(ste)
    axes[2].set_title("Short-Time Energy")

    zcr = zero_crossing_rate(signal, frame_length, hop_length)
    axes[3].plot(zcr)
    axes[3].set_title("Zero Crossing Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_analysis.png"), dpi=120)
    plt.close(fig)


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================
def extract_pitch_praat(path):
    """Pitch contour via Praat (parselmouth) - more reliable than librosa for this."""
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]  # drop unvoiced frames
    if len(f0_values) == 0:
        return 0.0
    return float(np.mean(f0_values))


def extract_formants_praat(path):
    """Mean F1, F2, F3 via Praat Burg method."""
    snd = parselmouth.Sound(path)
    formant = snd.to_formant_burg()
    duration = snd.get_total_duration()
    times = np.linspace(0.05, duration - 0.05, 20) if duration > 0.1 else [duration / 2]

    f1s, f2s, f3s = [], [], []
    for t in times:
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
        if not np.isnan(f1): f1s.append(f1)
        if not np.isnan(f2): f2s.append(f2)
        if not np.isnan(f3): f3s.append(f3)

    return {
        "F1": float(np.mean(f1s)) if f1s else 0.0,
        "F2": float(np.mean(f2s)) if f2s else 0.0,
        "F3": float(np.mean(f3s)) if f3s else 0.0,
    }


def extract_features(path, sr=SR):
    """Full feature set for one recording."""
    signal, sr = load_and_preprocess(path, sr=sr)
    frame_length, hop_length = frame_signal(signal, sr)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    energy = short_time_energy(signal, frame_length, hop_length)
    duration = librosa.get_duration(y=signal, sr=sr)
    pitch_mean = extract_pitch_praat(path)
    formants = extract_formants_praat(path)

    return {
        "mfcc": mfcc,                       # shape (12, T) -> used with DTW
        "mfcc_mean": np.mean(mfcc, axis=1),  # shape (12,) -> used with cosine/euclidean
        "energy_mean": float(np.mean(energy)) if len(energy) else 0.0,
        "duration": float(duration),
        "pitch_mean": pitch_mean,
        "formants": formants,
        "signal": signal,
        "sr": sr,
        "frame_length": frame_length,
        "hop_length": hop_length,
    }


# ============================================================
# 4. REFERENCE COMPARISON
# ============================================================
def mfcc_similarity(ref_mfcc, test_mfcc):
    alignment = dtw(ref_mfcc.T, test_mfcc.T, distance_only=True)
    raw_distance = alignment.distance
    similarity = 1.0 / (1.0 + raw_distance / 3000.0)  # normalize 0-1
    return float(np.clip(similarity, 0, 1)), float(raw_distance)


def cosine_similarity_score(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return float(1 - cosine(vec1, vec2))


def euclidean_similarity_score(vec1, vec2, scale=50.0):
    dist = euclidean(vec1, vec2)
    return float(1.0 / (1.0 + dist / scale))


def feature_diff_similarity(ref_val, test_val, scale):
    """Generic similarity for scalar features (pitch, duration): closer -> higher score."""
    if ref_val == 0:
        return 0.0
    diff = abs(ref_val - test_val) / ref_val
    return float(np.clip(1.0 - diff, 0, 1))


def formant_similarity(ref_formants, test_formants):
    sims = []
    for key in ("F1", "F2", "F3"):
        sims.append(feature_diff_similarity(ref_formants[key], test_formants[key], scale=None))
    return float(np.mean(sims))


# ============================================================
# 5. PRONUNCIATION SCORING
# ============================================================
def compute_pronunciation_score(ref_feats, test_feats):
    mfcc_sim, dtw_dist = mfcc_similarity(ref_feats["mfcc"], test_feats["mfcc"])
    pitch_sim = feature_diff_similarity(ref_feats["pitch_mean"], test_feats["pitch_mean"], scale=None)
    duration_sim = feature_diff_similarity(ref_feats["duration"], test_feats["duration"], scale=None)
    formant_sim = formant_similarity(ref_feats["formants"], test_feats["formants"])

    final_score = (
        WEIGHTS["mfcc"] * mfcc_sim +
        WEIGHTS["pitch"] * pitch_sim +
        WEIGHTS["duration"] * duration_sim +
        WEIGHTS["formant"] * formant_sim
    ) * 100

    breakdown = {
        "mfcc_similarity": round(mfcc_sim, 3),
        "dtw_distance": round(dtw_dist, 2),
        "pitch_similarity": round(pitch_sim, 3),
        "duration_similarity": round(duration_sim, 3),
        "formant_similarity": round(formant_sim, 3),
        "final_score": round(final_score, 1),
    }
    return breakdown


def feedback_from_score(breakdown):
    score = breakdown["final_score"]
    if score >= 85:
        quality = "ممتاز / Excellent"
    elif score >= 70:
        quality = "جيد / Good"
    elif score >= 50:
        quality = "متوسط / Needs improvement"
    else:
        quality = "ضعيف / Poor"

    notes = []
    if breakdown["formant_similarity"] < 0.6:
        notes.append("مشكلة في مكان النطق (formants) - الصوت غير متطابق مع المرجع في تردد الفورمنتس.")
    if breakdown["pitch_similarity"] < 0.6:
        notes.append("تفاوت واضح في طبقة الصوت (pitch).")
    if breakdown["duration_similarity"] < 0.6:
        notes.append("مدة النطق مختلفة بشكل ملحوظ عن المرجع (سريع/بطيء جدًا).")
    if breakdown["mfcc_similarity"] < 0.6:
        notes.append("الطابع الصوتي العام (MFCC) مختلف بشكل كبير عن النطق الصحيح.")
    if not notes:
        notes.append("النطق قريب جدًا من المرجع.")

    return quality, notes


# ============================================================
# 6. RUN THE THREE REQUIRED EXPERIMENTS
# ============================================================
import unicodedata

def find_recordings(data_dir):
    records = {}
    for speaker_dir in glob.glob(os.path.join(data_dir, "*")):
        if not os.path.isdir(speaker_dir):
            continue
        speaker = os.path.basename(speaker_dir)
        label = "incorrect" if "incorrect" in speaker.lower() else "correct"
        records[speaker] = {}
        for fname_raw in os.listdir(speaker_dir):
            if not fname_raw.lower().endswith(".wav"):
                continue
            f = os.path.join(speaker_dir, fname_raw)
            fname = fname_raw.rsplit(".", 1)[0]

            if "-" in fname:
                parts = [p.strip() for p in fname.split("-")]
                non_numeric = [p for p in parts if p and not p.isdigit()]
                word = non_numeric[0] if non_numeric else fname
            else:
                word = fname

            word = unicodedata.normalize("NFC", word.strip())
            records[speaker].setdefault(word, {})[label] = f
    return records


def print_records_debug(records):
    """Diagnostic helper - shows exactly what was found per speaker/word."""
    print("\n=== DEBUG: Recordings found ===")
    for speaker, word_map in records.items():
        print(f"\n[{speaker}]  ({len(word_map)} words)")
        for word, label_map in word_map.items():
            print(f"   '{word}'  -> {list(label_map.keys())}")
    print("=== END DEBUG ===\n")


def run_all_experiments(data_dir=DATA_DIR, reference_speaker=REFERENCE_SPEAKER):
    records = find_recordings(data_dir)
    if reference_speaker not in records:
        raise ValueError(f"Reference speaker '{reference_speaker}' not found in {data_dir}")

    results = []
    ref_lower = reference_speaker.lower()

    # build ref_feats for every reference word once (reused below for both the
    # original 3 experiments AND the new word-mismatch control experiments)
    ref_feats_by_word = {}
    for word, label_map in records[reference_speaker].items():
        if "correct" not in label_map:
            continue
        ref_feats_by_word[word] = extract_features(label_map["correct"])
        rf = ref_feats_by_word[word]
        plot_short_time_analysis(rf["signal"], rf["sr"], rf["frame_length"], rf["hop_length"],
                                  tag=f"REF_{reference_speaker}_{word}")

    word_list = sorted(ref_feats_by_word.keys())
    # cyclic mismatch pairing: each word is compared against the NEXT word in the
    # list as its "wrong word" control (e.g. "ضوء" mismatched against "قلم")
    mismatch_word = {w: word_list[(i + 1) % len(word_list)] for i, w in enumerate(word_list)}

    # ------------------------------------------------------------------
    # PART 1: original three required experiments (same word, correct ref)
    # ------------------------------------------------------------------
    for word, ref_feats in ref_feats_by_word.items():
        for speaker, word_map in records.items():
            if word not in word_map:
                continue
            for label, path in word_map[word].items():
                if speaker == reference_speaker and label == "correct":
                    continue  # skip comparing reference to itself

                test_feats = extract_features(path)
                breakdown = compute_pronunciation_score(ref_feats, test_feats)
                quality, notes = feedback_from_score(breakdown)

                speaker_lower = speaker.lower()
                if label == "incorrect":
                    experiment_type = "correct_vs_incorrect"
                elif speaker_lower == ref_lower or speaker_lower.startswith(ref_lower):
                    experiment_type = "same_speaker_same_word"
                else:
                    experiment_type = "different_speaker_same_word"

                results.append({
                    "word": word,
                    "mismatched_word": "",
                    "reference_speaker": reference_speaker,
                    "test_speaker": speaker,
                    "label": label,
                    "experiment_type": experiment_type,
                    **breakdown,
                    "quality": quality,
                    "notes": " | ".join(notes),
                })

    # ------------------------------------------------------------------
    # PART 2: control experiments - WRONG word on purpose (sanity check)
    # Only uses "correct" recordings, excludes the incorrect-pronunciation
    # speakers, to isolate the "word content" effect from "pronunciation
    # correctness" effect.
    # ------------------------------------------------------------------
    for word, ref_feats in ref_feats_by_word.items():
        other_word = mismatch_word[word]

        # --- Same speaker, different word (within the reference speaker's own folder) ---
        if other_word in records[reference_speaker] and "correct" in records[reference_speaker][other_word]:
            path = records[reference_speaker][other_word]["correct"]
            test_feats = extract_features(path)
            breakdown = compute_pronunciation_score(ref_feats, test_feats)
            quality, notes = feedback_from_score(breakdown)
            results.append({
                "word": word,
                "mismatched_word": other_word,
                "reference_speaker": reference_speaker,
                "test_speaker": reference_speaker,
                "label": "correct",
                "experiment_type": "same_speaker_different_word",
                **breakdown,
                "quality": quality,
                "notes": " | ".join(notes),
            })

        # --- Different speaker, different word ---
        for speaker, word_map in records.items():
            speaker_lower = speaker.lower()
            if speaker_lower == ref_lower or speaker_lower.startswith(ref_lower):
                continue  # same physical person, not a genuinely different speaker
            if "incorrect" in speaker_lower:
                continue  # keep this control clean: correctness is a separate dimension
            if other_word in word_map and "correct" in word_map[other_word]:
                path = word_map[other_word]["correct"]
                test_feats = extract_features(path)
                breakdown = compute_pronunciation_score(ref_feats, test_feats)
                quality, notes = feedback_from_score(breakdown)
                results.append({
                    "word": word,
                    "mismatched_word": other_word,
                    "reference_speaker": reference_speaker,
                    "test_speaker": speaker,
                    "label": "correct",
                    "experiment_type": "different_speaker_different_word",
                    **breakdown,
                    "quality": quality,
                    "notes": " | ".join(notes),
                })

    df = pd.DataFrame(results)
    df.to_csv("pronunciation_results.csv", index=False, encoding="utf-8-sig")
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Running Arabic Pronunciation Assessment pipeline...")

    records_preview = find_recordings(DATA_DIR)
    print_records_debug(records_preview)

    df = run_all_experiments()
    if df.empty:
        print("⚠️  NO RESULTS - the dataframe is empty.")
        print("Check the DEBUG output above: word names must match EXACTLY across")
        print("all speaker folders (same Arabic spelling, no extra characters).")
    else:
        print(df[["word", "test_speaker", "label", "experiment_type", "final_score", "quality"]])
        print("\nFull results saved to pronunciation_results.csv")
        print(f"Plots saved to ./{PLOTS_DIR}/")

        print("\n--- Summary by experiment type ---")
        print(df.groupby("experiment_type")["final_score"].agg(["mean", "std", "count"]))