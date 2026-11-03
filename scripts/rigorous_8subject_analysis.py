"""
Script: scripts/rigorous_8subject_analysis.py

Addresses:
  Q2: Recompute D_lip(t) summary stat Cohen's d and p-values for the 8-subject offline dataset
      using subject-level means (N=8 per condition) vs the prior session-level (N=24).
      Report both versions side by side.

  Q3: Leave-One-Identity-Out (LOIO) cross-validation across all 8 subjects
      (8 folds, each holding out one identity's 3 sessions), using Logistic Regression
      on D_lip(t) kinematics. Report per-fold and averaged accuracy, precision, recall, AUC.
"""

import sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.lip_tracker import LipTracker

BASE_DIR = Path("data/raw/live_sessions")
FEAT_NAMES = ["mean", "std", "min", "max", "roc"]


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

    ap = np.array(apertures, dtype=np.float64)
    vel = np.abs(np.diff(ap))
    return {
        "mean": float(np.mean(ap)),
        "std":  float(np.std(ap, ddof=1)) if len(ap) > 1 else 0.0,
        "min":  float(np.min(ap)),
        "max":  float(np.max(ap)),
        "roc":  float(np.mean(vel)) if len(vel) > 0 else 0.0,
    }


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(y) - np.mean(x)) / sp if sp > 1e-9 else 0.0


def main():
    tracker = LipTracker()

    # ---- Extract all 48 sessions with subject identity tagged ----
    records = []  # {subj, cond, label, session, feat_dict}
    for s_idx in range(1, 9):
        subj = f"S{s_idx}"
        for cond, label in [("genuine", 0), ("swap", 1)]:
            ext = ".mov" if cond == "genuine" else ".mp4"
            dir_p = BASE_DIR / subj / cond
            for sess in [1, 2, 3]:
                fp = dir_p / f"session_{sess}{ext}"
                if fp.exists():
                    feat = extract_clip_features(tracker, fp)
                    if feat is not None:
                        records.append({"subj": subj, "cond": cond, "label": label,
                                        "session": sess, **feat})

    n_clips = len(records)
    print(f"\n[*] Extracted features from {n_clips} clips across 8 subjects.")

    # =====================================================================
    # Q2 — SIDE-BY-SIDE SESSION-LEVEL vs SUBJECT-LEVEL COHEN'S D
    # =====================================================================
    print("\n" + "=" * 105)
    print(f"{'Q2 — SESSION-LEVEL (N=24 per condition) vs SUBJECT-LEVEL (N=8 per condition) EFFECT SIZES':^105}")
    print("=" * 105)

    # Session-level arrays (N up to 24 per condition)
    sess_real = {f: [r[f] for r in records if r["label"] == 0] for f in FEAT_NAMES}
    sess_swap = {f: [r[f] for r in records if r["label"] == 1] for f in FEAT_NAMES}

    # Subject-level: average each feature per subject per condition
    subj_real_means = defaultdict(lambda: defaultdict(list))
    subj_swap_means = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["label"] == 0:
            for f in FEAT_NAMES:
                subj_real_means[r["subj"]][f].append(r[f])
        else:
            for f in FEAT_NAMES:
                subj_swap_means[r["subj"]][f].append(r[f])

    subj_real = {f: [np.mean(subj_real_means[s][f]) for s in sorted(subj_real_means)] for f in FEAT_NAMES}
    subj_swap = {f: [np.mean(subj_swap_means[s][f]) for s in sorted(subj_swap_means)] for f in FEAT_NAMES}

    print(f"\n{'Feature':<8} | {'Session d':>9} | {'Session p':>9} | {'Subject d':>9} | {'Subject p':>9} | Interpretation")
    print("-" * 105)
    for f in FEAT_NAMES:
        sr, ss = np.array(sess_real[f]), np.array(sess_swap[f])
        ur, us = np.array(subj_real[f]), np.array(subj_swap[f])

        d_sess = cohens_d(sr, ss)
        _, p_sess = stats.ttest_ind(sr, ss)
        d_subj = cohens_d(ur, us)
        _, p_subj = stats.ttest_ind(ur, us)

        if abs(d_subj) >= 0.80:
            interp = "Large effect (≥0.80)"
        elif abs(d_subj) >= 0.50:
            interp = "Moderate effect (0.50–0.79)"
        elif abs(d_subj) >= 0.20:
            interp = "Small effect (0.20–0.49)"
        else:
            interp = "Negligible (<0.20)"

        print(f"{f:<8} | {d_sess:>+9.3f} | {p_sess:>9.4f} | {d_subj:>+9.3f} | {p_subj:>9.4f} | {interp}")
    print("-" * 105)
    print(f"  N (session-level): {len(sess_real['mean'])} Real, {len(sess_swap['mean'])} Swap")
    print(f"  N (subject-level): {len(subj_real['mean'])} Real subjects, {len(subj_swap['mean'])} Swap subjects")

    # =====================================================================
    # Q3 — LEAVE-ONE-IDENTITY-OUT CROSS-VALIDATION (8 folds)
    # =====================================================================
    print("\n" + "=" * 105)
    print(f"{'Q3 — LEAVE-ONE-IDENTITY-OUT CROSS-VALIDATION (8 FOLDS)':^105}")
    print("=" * 105)
    print(f"  Model: Logistic Regression (C=1.0, StandardScaler, random_state=42)")
    print(f"  Feature set: mean, std, min, max, rate-of-change of D_lip(t)")
    print()

    subjects = [f"S{i}" for i in range(1, 9)]
    fold_results = []

    print(f"{'Fold (Test)':<14} | {'Test N':>6} | {'Acc %':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'AUC':>6} | {'CM [TN,FP,FN,TP]'}")
    print("-" * 105)

    for held_out in subjects:
        train_recs = [r for r in records if r["subj"] != held_out]
        test_recs  = [r for r in records if r["subj"] == held_out]

        if not train_recs or not test_recs:
            continue

        X_tr = np.array([[r[f] for f in FEAT_NAMES] for r in train_recs])
        y_tr = np.array([r["label"] for r in train_recs])
        X_te = np.array([[r[f] for f in FEAT_NAMES] for r in test_recs])
        y_te = np.array([r["label"] for r in test_recs])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(X_tr_s, y_tr)

        preds = clf.predict(X_te_s)
        probs = clf.predict_proba(X_te_s)[:, 1]

        acc   = accuracy_score(y_te, preds) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = float("nan")

        tn = int(np.sum((y_te == 0) & (preds == 0)))
        fp = int(np.sum((y_te == 0) & (preds == 1)))
        fn = int(np.sum((y_te == 1) & (preds == 0)))
        tp = int(np.sum((y_te == 1) & (preds == 1)))

        fold_results.append({
            "fold": held_out, "n_test": len(test_recs),
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
        print(f"{held_out:<14} | {len(test_recs):>6} | {acc:>5.1f}% | {prec:>6.4f} | {rec:>6.4f} | {f1:>6.4f} | {auc:>6.4f} | [{tn},{fp},{fn},{tp}]")

    print("-" * 105)

    # Averaged across folds
    avg_acc  = np.mean([x["acc"]  for x in fold_results])
    avg_prec = np.mean([x["prec"] for x in fold_results])
    avg_rec  = np.mean([x["rec"]  for x in fold_results])
    avg_f1   = np.mean([x["f1"]   for x in fold_results])
    valid_aucs = [x["auc"] for x in fold_results if not np.isnan(x["auc"])]
    avg_auc  = np.mean(valid_aucs) if valid_aucs else float("nan")
    std_acc  = np.std([x["acc"] for x in fold_results], ddof=1)
    std_auc  = np.std(valid_aucs, ddof=1) if len(valid_aucs) > 1 else float("nan")
    total_tn = sum(x["tn"] for x in fold_results)
    total_fp = sum(x["fp"] for x in fold_results)
    total_fn = sum(x["fn"] for x in fold_results)
    total_tp = sum(x["tp"] for x in fold_results)

    print(f"{'LOIO AVERAGE':<14} | {'48':>6} | {avg_acc:>5.1f}% | {avg_prec:>6.4f} | {avg_rec:>6.4f} | {avg_f1:>6.4f} | {avg_auc:>6.4f} | Aggregate [{total_tn},{total_fp},{total_fn},{total_tp}]")
    print(f"{'(± std)':<14} | {'':>6} | {std_acc:>5.1f}% | {'':>6} | {'':>6} | {'':>6} | {std_auc:>6.4f} |")

    print("\n" + "=" * 105)
    print(f"{'LOIO CONCLUSION':^105}")
    print("=" * 105)
    majority_baseline = 50.0
    if avg_acc > majority_baseline and avg_auc > 0.55:
        print(f"  Mean LOIO accuracy = {avg_acc:.1f}% (baseline {majority_baseline:.1f}%) | Mean AUC = {avg_auc:.4f}")
        print(f"  The visual kinematic branch (D_lip summary stats) generalises across unseen identities.")
    else:
        print(f"  Mean LOIO accuracy = {avg_acc:.1f}% (baseline {majority_baseline:.1f}%) | Mean AUC = {avg_auc:.4f}")
        print(f"  The visual kinematic branch does NOT consistently beat baseline across unseen identities.")
    print("=" * 105)


if __name__ == "__main__":
    main()
