"""
Script: scripts/falsify_8subject_visual_signal.py
Signal-Existence Falsification Test on the 8-Subject Dataset (48 clips).
Extracts D_lip(t) kinematics summary statistics (mean, std, min, max, rate_of_change),
fits a Logistic Regression classifier on disjoint identity graph splits (Train: S1-S5, Test: S7-S8),
and evaluates whether visual landmark statistics beat the majority-class baseline (50.0%).
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.lip_tracker import LipTracker

BASE_DIR = Path("data/raw/live_sessions")


def extract_clip_features(tracker: LipTracker, file_path: Path):
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_dur = (1.0 / fps) * 1000.0

    apertures = []
    f_idx = 0
    while cap.isOpened() and f_idx < 200:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        res = tracker.process_frame(frame, timestamp_ms=f_idx * frame_dur)
        if res.face_detected and res.lip_aperture > 0:
            apertures.append(res.lip_aperture)
        f_idx += 1
    cap.release()

    if len(apertures) < 5:
        return None

    ap_arr = np.array(apertures, dtype=np.float64)
    # Rate of change: frame-to-frame absolute delta
    velocity = np.abs(np.diff(ap_arr))
    
    mean_ap = np.mean(ap_arr)
    std_ap = np.std(ap_arr, ddof=1) if len(ap_arr) > 1 else 0.0
    min_ap = np.min(ap_arr)
    max_ap = np.max(ap_arr)
    roc_ap = np.mean(velocity) if len(velocity) > 0 else 0.0

    return {
        "mean": mean_ap,
        "std": std_ap,
        "min": min_ap,
        "max": max_ap,
        "roc": roc_ap,
        "frames_valid": len(apertures),
    }


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(y) - np.mean(x)) / sp if sp > 1e-9 else 0.0


def main():
    print("=" * 85)
    print(f"{'8-SUBJECT VISUAL SIGNAL FALSIFICATION TEST (N=48 CLIPS)':^85}")
    print("=" * 85)

    tracker = LipTracker()

    records = []
    for s_idx in range(1, 9):
        subj = f"S{s_idx}"
        for cond, label in [("genuine", 0), ("swap", 1)]:
            dir_p = BASE_DIR / subj / cond
            ext = ".mov" if cond == "genuine" else ".mp4"
            for sess in [1, 2, 3]:
                fp = dir_p / f"session_{sess}{ext}"
                if fp.exists():
                    feat = extract_clip_features(tracker, fp)
                    if feat is not None:
                        records.append({
                            "subj": subj,
                            "cond": cond,
                            "label": label,
                            "path": str(fp),
                            **feat
                        })

    print(f"[*] Successfully extracted features for {len(records)} / 48 clips.\n")

    # Feature Matrix
    feat_names = ["mean", "std", "min", "max", "roc"]
    
    # 1. Feature Distribution Summary: Real vs Swap
    real_feats = {f: [r[f] for r in records if r["label"] == 0] for f in feat_names}
    swap_feats = {f: [r[f] for r in records if r["label"] == 1] for f in feat_names}

    print("-" * 85)
    print(f"{'Feature':<12} | {'Real Mean ± Std':<20} | {'Swap Mean ± Std':<20} | {'Cohen d':<8} | {'p-value':<10}")
    print("-" * 85)
    for f in feat_names:
        r_vals = np.array(real_feats[f])
        s_vals = np.array(swap_feats[f])
        d = cohens_d(r_vals, s_vals)
        t_stat, p_val = stats.ttest_ind(r_vals, s_vals)
        print(f"{f:<12} | {np.mean(r_vals):.4f} ± {np.std(r_vals, ddof=1):.4f}      | {np.mean(s_vals):.4f} ± {np.std(s_vals, ddof=1):.4f}      | {d:>7.3f} | {p_val:>9.4f}")
    print("-" * 85)

    # 2. Identity-Disjoint Splitting
    train_ids = {"S1", "S2", "S3", "S4", "S5"}
    val_ids = {"S6"}
    test_ids = {"S7", "S8"}

    train_recs = [r for r in records if r["subj"] in train_ids]
    val_recs = [r for r in records if r["subj"] in val_ids]
    test_recs = [r for r in records if r["subj"] in test_ids]

    X_train = np.array([[r[f] for f in feat_names] for r in train_recs])
    y_train = np.array([r["label"] for r in train_recs])

    X_test = np.array([[r[f] for f in feat_names] for r in test_recs])
    y_test = np.array([r["label"] for r in test_recs])

    print(f"\n[*] Disjoint Split Counts:")
    print(f"  • Train (S1-S5): {len(train_recs)} clips ({sum(y_train == 0)} Real, {sum(y_train == 1)} Swap)")
    print(f"  • Test  (S7-S8): {len(test_recs)} clips ({sum(y_test == 0)} Real, {sum(y_test == 1)} Swap)")

    # 3. Logistic Regression Classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_train_scaled, y_train)

    train_preds = clf.predict(X_train_scaled)
    test_preds = clf.predict(X_test_scaled)
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, train_preds) * 100.0
    test_acc = accuracy_score(y_test, test_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, test_probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, test_preds)

    print("\n" + "=" * 85)
    print(f"{'EVALUATION ON HELD-OUT TEST SPLIT (SUBJECTS S7 & S8)':^85}")
    print("=" * 85)
    print(f"  • Majority-Class Baseline: 50.00% (Balanced 6 Real / 6 Swap)")
    print(f"  • Train Accuracy:          {train_acc:.2f}%")
    print(f"  • Test Accuracy:           {test_acc:.2f}%")
    print(f"  • Test Precision:          {prec:.4f}")
    print(f"  • Test Recall:             {rec:.4f}")
    print(f"  • Test F1-Score:           {f1:.4f}")
    print(f"  • Test ROC-AUC Score:      {auc:.4f}")
    print(f"  • Confusion Matrix [TN, FP / FN, TP]:\n    {cm.tolist()}")
    print(f"  • Model Learned Coefficients: {dict(zip(feat_names, clf.coef_[0].round(4)))}")

    print("\n" + "=" * 85)
    print("FALSIFICATION CONCLUSION:")
    if test_acc <= 50.0 or auc <= 0.55:
        print("  [X] FALSIFICATION CONFIRMED: D_lip(t) kinematics summary statistics CANNOT separate")
        print("      offline 2D face-swaps from genuine takes on unseen identities.")
        print("      Test accuracy ({}%) fails to beat the 50.0% majority baseline (ROC-AUC={:.4f}).".format(test_acc, auc))
        print("      Conclusion: Offline 2D face-swaps preserve driving kinematics; the visual aperture")
        print("      channel alone contains NO statistically separable deepfake signal on unseen identities.")
    else:
        print("  [✓] Signal Detected: Test accuracy ({:.2f}%) exceeds baseline with AUC={:.4f}.".format(test_acc, auc))
    print("=" * 85)


if __name__ == "__main__":
    main()
