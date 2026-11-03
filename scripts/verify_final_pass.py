#!/usr/bin/env python3
"""
Final Verification Script for SecureKYC AI.
Computes and prints:
1. Deployed Rule evaluation on held-out test split (raw code output)
2. Pytest test count
3. PDF page counts
4. Abstract word count
5. First page text of both IEEE PDFs (Camera-Ready & Anonymized)
"""

import sys
import re
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR, LIP_CLOSURE_THRESHOLD_NORMALIZED
from src.models.fusion_net import VisionOnlyNet, DesyncOnlyNet
from src.models.train import split_dataset_by_identity

def main():
    print("=" * 80)
    print(" 1. DEPLOYED RULE EVALUATION ON HELD-OUT TEST SPLIT (N=103) ")
    print("=" * 80)

    device = torch.device("cpu")
    data_path = PROCESSED_DATA_DIR / "dataset.pt"
    data = torch.load(data_path, map_location=device, weights_only=False)
    train_dict, val_dict, test_dict = split_dataset_by_identity(data, random_seed=42)

    lip_test = test_dict["lip_sequences"].to(device)
    desync_test = test_dict["desync_features"].to(device)
    y_test = test_dict["labels"].cpu().numpy().flatten().astype(int)

    ckpt_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vis_net = VisionOnlyNet(seq_len=100).to(device)
    vis_net.load_state_dict(ckpt["vision_state_dict"])
    vis_net.eval()

    desync_net = DesyncOnlyNet(desync_feature_dim=3).to(device)
    desync_net.load_state_dict(ckpt["desync_state_dict"])
    desync_net.eval()

    with torch.no_grad():
        p_vis = vis_net(lip_test).cpu().numpy().flatten()
        p_des = desync_net(desync_features=desync_test).cpu().numpy().flatten()

    w_vis = float(ckpt["meta_weights"]["w_vis"])
    w_des = float(ckpt["meta_weights"]["w_desync"])
    b = float(ckpt["meta_weights"]["intercept"])
    z = w_vis * p_vis + w_des * p_des + b
    p_fused = expit(z)

    s_desync = desync_test[:, 0].cpu().numpy()

    # Rule deployed in app/main.py: (p_fused >= 0.5) OR (S_desync > LIP_CLOSURE_THRESHOLD_NORMALIZED)
    preds_deployed = ((p_fused >= 0.5) | (s_desync > LIP_CLOSURE_THRESHOLD_NORMALIZED)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds_deployed, labels=[0, 1]).ravel()
    acc = accuracy_score(y_test, preds_deployed)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds_deployed, average="binary", zero_division=0)
    apcer = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    bpcer = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"Total Test Samples: {len(y_test)} (Authentic Genuine={int((y_test==0).sum())}, Manipulated Attack={int((y_test==1).sum())})")
    print(f"Calibrated Physical Closure Threshold: LIP_CLOSURE_THRESHOLD_NORMALIZED = {LIP_CLOSURE_THRESHOLD_NORMALIZED}")
    print(f"Rule: (p_fused >= 0.5) OR (S_desync > {LIP_CLOSURE_THRESHOLD_NORMALIZED})")
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score : {f1:.4f}")
    print(f"APCER    : {apcer:.4f} ({apcer*100:.2f}%)")
    print(f"BPCER    : {bpcer:.4f} ({bpcer*100:.2f}%)")

    print("\n" + "=" * 80)
    print(" 2. ABSTRACT WORD COUNT ")
    print("=" * 80)
    md_paper = (PROJECT_ROOT / "paper" / "ieee_research_paper.md").read_text(encoding="utf-8")
    m_abs = re.search(r"###\s+Abstract\s*\n+(.*?)(?=\n+\*\*Index Terms\*\*|\n+##\s+I\.)", md_paper, re.DOTALL)
    if m_abs:
        abs_text = m_abs.group(1).strip()
        words = abs_text.split()
        print(f"Abstract Word Count: {len(words)} words")
    else:
        print("Abstract not found via regex!")

    print("\n" + "=" * 80)
    print(" 3. PDF PAGE COUNTS ")
    print("=" * 80)
    pdf_paths = [
        PROJECT_ROOT / "paper" / "SecureKYC_AI_IEEE_Research_Paper.pdf",
        PROJECT_ROOT / "paper" / "SecureKYC_AI_IEEE_Research_Paper_Anonymized.pdf",
        PROJECT_ROOT / "paper" / "SecureKYC_AI_Review_2_Report.pdf",
    ]
    for p in pdf_paths:
        r = PdfReader(p)
        print(f"{p.name:<48} : {len(r.pages)} pages")

    print("\n" + "=" * 80)
    print(" 4. FIRST PAGE TEXT: CAMERA-READY IEEE PDF ")
    print("=" * 80)
    r_cam = PdfReader(pdf_paths[0])
    print(r_cam.pages[0].extract_text()[:600])

    print("\n" + "=" * 80)
    print(" 5. FIRST PAGE TEXT: ANONYMIZED IEEE PDF ")
    print("=" * 80)
    r_anon = PdfReader(pdf_paths[1])
    print(r_anon.pages[0].extract_text()[:600])

if __name__ == "__main__":
    main()
