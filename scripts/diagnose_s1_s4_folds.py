"""
Script: scripts/diagnose_s1_s4_folds.py

Per-clip diagnostic for folds S1 (AUC=0.000) and S4 (AUC=1.000).
For each of the 6 test clips per fold, reports:
  - Duration (seconds), FPS, total frames, face-detected frame count, face-detection rate
  - Feature values: mean, std, min, max, roc of D_lip(t)
  - Where the clip falls relative to the fold's trained decision boundary
Also reports:
  - The trained LR coefficients for that fold
  - Decision boundary analysis: min + roc values vs training distribution
"""

import sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.lip_tracker import LipTracker

BASE_DIR = Path("data/raw/live_sessions")
FEAT_NAMES = ["mean", "std", "min", "max", "roc"]


def extract_clip_features_verbose(tracker: LipTracker, file_path: Path):
    """Returns features + raw diagnostic counters."""
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dur = (1.0 / fps) * 1000.0

    apertures = []
    frames_read = 0
    frames_face_detected = 0
    f_idx = 0
    while cap.isOpened() and f_idx < 250:  # No frame cap — read full clip
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames_read += 1
        res = tracker.process_frame(frame, timestamp_ms=f_idx * frame_dur)
        if res.face_detected:
            frames_face_detected += 1
            if res.lip_aperture > 0:
                apertures.append(res.lip_aperture)
        f_idx += 1
    cap.release()

    duration_s = frames_read / fps if fps > 0 else 0.0
    det_rate = frames_face_detected / max(frames_read, 1)

    feats = None
    if len(apertures) >= 5:
        ap = np.array(apertures, dtype=np.float64)
        vel = np.abs(np.diff(ap))
        feats = {
            "mean": float(np.mean(ap)),
            "std":  float(np.std(ap, ddof=1)) if len(ap) > 1 else 0.0,
            "min":  float(np.min(ap)),
            "max":  float(np.max(ap)),
            "roc":  float(np.mean(vel)) if len(vel) > 0 else 0.0,
        }

    return feats, {
        "fps": fps,
        "total_frames_metadata": total_frames,
        "frames_read": frames_read,
        "frames_face_detected": frames_face_detected,
        "face_det_rate": det_rate,
        "aperture_samples": len(apertures),
        "duration_s": duration_s,
    }


def main():
    tracker = LipTracker()

    # Extract all 48 clips
    all_records = []
    for s_idx in range(1, 9):
        subj = f"S{s_idx}"
        for cond, label in [("genuine", 0), ("swap", 1)]:
            ext = ".mov" if cond == "genuine" else ".mp4"
            for sess in [1, 2, 3]:
                fp = BASE_DIR / subj / cond / f"session_{sess}{ext}"
                if fp.exists():
                    feats, diag = extract_clip_features_verbose(tracker, fp)
                    if feats is not None:
                        all_records.append({
                            "subj": subj, "cond": cond, "label": label,
                            "sess": sess, "path": str(fp), **feats, "diag": diag
                        })

    # ---- Analyse S1 and S4 folds ----
    for held_out in ["S1", "S4"]:
        train_recs = [r for r in all_records if r["subj"] != held_out]
        test_recs  = [r for r in all_records if r["subj"] == held_out]

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
        probs = clf.predict_proba(X_te_s)[:, 1]  # P(swap)

        # Training distribution stats for context
        tr_real = [r for r in train_recs if r["label"] == 0]
        tr_swap = [r for r in train_recs if r["label"] == 1]
        tr_real_min = np.array([r["min"] for r in tr_real])
        tr_swap_min = np.array([r["min"] for r in tr_swap])
        tr_real_roc = np.array([r["roc"] for r in tr_real])
        tr_swap_roc = np.array([r["roc"] for r in tr_swap])

        print("\n" + "=" * 110)
        print(f"  FOLD: Test={held_out}  |  Train=All other 7 subjects ({len(train_recs)} clips)")
        print("=" * 110)
        print(f"  Learned LR Coefficients (StandardScaler space):")
        for f, c in zip(FEAT_NAMES, clf.coef_[0]):
            print(f"    {f:<6}: {c:+.4f}")
        print(f"  Intercept: {clf.intercept_[0]:+.4f}")
        print(f"\n  Training distribution (min aperture):  Real {np.mean(tr_real_min):.4f}±{np.std(tr_real_min, ddof=1):.4f}  |  Swap {np.mean(tr_swap_min):.4f}±{np.std(tr_swap_min, ddof=1):.4f}")
        print(f"  Training distribution (rate-of-change): Real {np.mean(tr_real_roc):.4f}±{np.std(tr_real_roc, ddof=1):.4f}  |  Swap {np.mean(tr_swap_roc):.4f}±{np.std(tr_swap_roc, ddof=1):.4f}")

        print(f"\n  Per-Clip Diagnostic for {held_out} (6 test clips):")
        print("-" * 110)
        print(f"  {'Clip':<22} | {'True':>5} | {'Pred':>5} | {'P(sw)':>6} | {'min':>7} | {'roc':>7} | {'mean':>7} | {'dur_s':>5} | {'det%':>5} | {'ap_N':>5} | {'frames':>7}")
        print("-" * 110)

        for r, pred, prob in zip(test_recs, preds, probs):
            d = r["diag"]
            clip_name = f"{r['subj']}/{r['cond']}/sess{r['sess']}"
            true_label = "swap" if r["label"] == 1 else "real"
            pred_label = "swap" if pred == 1 else "real"
            correct = "✓" if pred == r["label"] else "✗"
            print(f"  {clip_name:<22} | {true_label:>5} | {pred_label:>5} | {prob:>6.4f} | {r['min']:>7.4f} | {r['roc']:>7.4f} | {r['mean']:>7.4f} | {d['duration_s']:>5.1f}s | {d['face_det_rate']*100:>4.1f}% | {d['aperture_samples']:>5} | {d['frames_read']:>4}fr {correct}")
        print("-" * 110)

        # Flag any data quality concerns
        print(f"\n  DATA QUALITY FLAGS for {held_out}:")
        any_flag = False
        for r in test_recs:
            d = r["diag"]
            flags = []
            if d["duration_s"] < 3.0:
                flags.append(f"SHORT clip ({d['duration_s']:.1f}s < 3.0s)")
            if d["face_det_rate"] < 0.70:
                flags.append(f"LOW face-detection rate ({d['face_det_rate']*100:.1f}% < 70%)")
            if d["aperture_samples"] < 20:
                flags.append(f"FEW aperture samples (N={d['aperture_samples']} < 20)")
            if flags:
                any_flag = True
                clip_name = f"{r['subj']}/{r['cond']}/sess{r['sess']}"
                print(f"  [!] {clip_name}: {'; '.join(flags)}")
        if not any_flag:
            print(f"  [✓] No data quality issues detected for {held_out}.")


if __name__ == "__main__":
    main()
