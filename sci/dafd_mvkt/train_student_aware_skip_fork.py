"""
Student-aware branch distillation for skip-fork structures.

Jointly fine-tunes t2, t3, and S while keeping t1 frozen.
Branches receive a weak KD signal from the current student.

Loss:
    L_t2 = BCE(t2) + lambda_kd * KD(t1→t2)
           + lambda_sa * KD(S.detach()→t2)
    L_t3 = BCE(t3) + lambda_kd * KD(t1→t3)
           + lambda_sa * KD(S.detach()→t3)
    L_S  = BCE(S) + lambda_kd * [w1*KD(t1→S)
                                 + w2*KD(t2.detach()→S)
                                 + w3*KD(t3.detach()→S)]

t1 is always frozen.  lambda_sa must be small (0.05 or 0.1).

Supported SF structures (via --sf_id):
    SF01: t1=12L100, t2=12L50,  t3=1L100  weights=[0.2, 0.4, 0.4]
    SF02: t1=12L100, t2=1L500,  t3=1L100  weights=[0.2, 0.4, 0.4]

Usage:
    python dafd_mvkt/train_student_aware_skip_fork.py \\
        --config            dafd_mvkt/configs/adaptive_50hz.yaml \\
        --data_dir          comper_repo/ptb_xl \\
        --sf_id             SF02 \\
        --t1_ckpt           dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt \\
        --t2_ckpt           dafd_mvkt/outputs/skipfork/student_II_50hz_skipfork_SF02_12L100_to_1L500_1L100_w020_040_040_seed0_t2_best.pt \\
        --t3_ckpt           dafd_mvkt/outputs/skipfork/student_II_50hz_skipfork_SF02_12L100_to_1L500_1L100_w020_040_040_seed0_t3_best.pt \\
        --student_ckpt      dafd_mvkt/outputs/skipfork/student_II_50hz_skipfork_SF02_12L100_to_1L500_1L100_w020_040_040_seed0_best.pt \\
        --lambda_sa         0.05 \\
        --seed              0
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from data.ptbxl_multiteacher_dataset import PTBXLMultiTeacherDataset, SUPERCLASSES
from losses.mkd import multi_label_kd_loss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import load_encoder_init

BCE_AUC  = 0.8061; BCE_F1  = 0.5740
PROG_AUC = 0.8396; PROG_F1 = 0.6176
C14_AUC  = 0.8425; C14_F1  = 0.6243
MVKT_AUC = 0.843;  MVKT_F1 = 0.626

VIEW_DEFS: dict[str, dict] = {
    "12L500": {"in_channels": 12, "input_key": "teacher_12l_500"},
    "12L100": {"in_channels": 12, "input_key": "teacher_12l_100"},
    "12L50":  {"in_channels": 12, "input_key": "teacher_12l_50"},
    "1L500":  {"in_channels":  1, "input_key": "teacher_1l_500"},
    "1L100":  {"in_channels":  1, "input_key": "teacher_1l_100"},
    "1L50":   {"in_channels":  1, "input_key": "teacher_1l_50"},
}

SF_STRUCTURES: dict[str, dict] = {
    "SF01": {
        "t1": "12L100", "t2": "12L50",  "t3": "1L100",
        "weights": [0.2, 0.4, 0.4],
    },
    "SF02": {
        "t1": "12L100", "t2": "1L500",  "t3": "1L100",
        "weights": [0.2, 0.4, 0.4],
    },
}

STUDENT_IN_CH = 1
STUDENT_KEY   = "student_x"   # 1L50 — II lead 50Hz


def _build_model(in_channels: int, num_classes: int = 5) -> ResNet1d:
    return ResNet1d(in_channels=in_channels, num_classes=num_classes,
                    layers=[3, 4, 6, 3], base_channels=64,
                    proj_dim=128, dropout=0.0)


@torch.no_grad()
def _run_eval(model: nn.Module, loader: DataLoader,
              input_key: str, device: torch.device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x   = batch[input_key].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels, all_ids


def _final_eval(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_key: str,
    device: torch.device,
    out_name: str,
    out_dir: Path,
    meta: dict,
) -> dict:
    val_probs, val_labels, _          = _run_eval(model, val_loader,  input_key, device)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(model, test_loader, input_key, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    metrics = {
        "output_name":    out_name,
        "macro_auc":      round(m05["macro_auc"], 5),
        "macro_f1_0_5":   round(m05.get("macro_f1", 0), 5),
        "macro_f1_tuned": round(m_tune.get("macro_f1", 0), 5),
        "per_class_thresholds": {SUPERCLASSES[i]: round(float(thresholds[i]), 4)
                                 for i in range(5)},
        "_meta": meta,
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    with open(out_dir / f"metrics_val_{out_name}.json", "w") as f:
        json.dump({"output_name": out_name,
                   "macro_auc":      round(val_m05["macro_auc"], 5),
                   "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5)}, f, indent=2)
    with open(out_dir / f"metrics_test_{out_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows  = [{"record_id": rid,
               **{f"y_{c}":    int(test_labels[i, ci])           for ci, c in enumerate(SUPERCLASSES)},
               **{f"prob_{c}": round(float(test_probs[i, ci]), 5) for ci, c in enumerate(SUPERCLASSES)},
               **{f"pred_{c}": int(preds[i, ci])                  for ci, c in enumerate(SUPERCLASSES)}}
             for i, rid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(out_dir / f"predictions_test_{out_name}.csv", index=False)
    return metrics


def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed if args.seed is not None else tc.get("seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ep    = args.epochs     or tc.get("epochs", 100)
    bs    = args.batch_size or tc.get("batch_size", 512)
    lr_   = tc.get("lr", 1e-3)
    nw    = tc.get("num_workers", 4)
    pw    = tc.get("persistent_workers", True)
    wd    = tc.get("weight_decay", 1e-4)
    gc    = tc.get("grad_clip", 1.0)

    lambda_kd   = args.lambda_kd   if args.lambda_kd   is not None else tc.get("lambda_kd",  1.0)
    lambda_sa   = args.lambda_sa   if args.lambda_sa   is not None else tc.get("lambda_sa",   0.05)
    temperature = args.temperature if args.temperature is not None else tc.get("temperature", 2.0)

    sf_id  = args.sf_id
    sf_def = SF_STRUCTURES[sf_id]
    t1_view, t2_view, t3_view = sf_def["t1"], sf_def["t2"], sf_def["t3"]
    w1, w2, w3 = sf_def["weights"]

    t1_def = VIEW_DEFS[t1_view]
    t2_def = VIEW_DEFS[t2_view]
    t3_def = VIEW_DEFS[t3_view]

    lam_str   = str(lambda_sa).replace(".", "")
    base_name = args.output_base or (
        f"student_II_50hz_saf_{sf_id.lower()}_{t1_view.lower()}_"
        f"to_{t2_view.lower()}_{t3_view.lower()}_sa{lam_str}_seed{seed}")
    s_out_name  = base_name
    t2_out_name = base_name + "_t2"
    t3_out_name = base_name + "_t3"

    out_dir = Path(args.output_dir or cfg.get("outputs", {}).get("dir", "dafd_mvkt/outputs/adaptive"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    s_ckpt  = out_dir / f"{s_out_name}_best.pt"
    t2_ckpt = out_dir / f"{t2_out_name}_best.pt"
    t3_ckpt = out_dir / f"{t3_out_name}_best.pt"

    print(f"\n{'='*60}")
    print(f"Student-Aware Skip-Fork: {sf_id}")
    print(f"  t1={t1_view}(frozen)  t2={t2_view}  t3={t3_view}")
    print(f"  weights=[{w1},{w2},{w3}]  lambda_sa={lambda_sa}  T={temperature}")
    print(f"  device={device}  seed={seed}  epochs={ep}  bs={bs}")
    print(f"  output: {s_ckpt}")
    print(f"{'='*60}")

    # ── frozen t1 ─────────────────────────────────────────────────────────────
    t1 = _build_model(t1_def["in_channels"]).to(device)
    load_checkpoint(args.t1_ckpt, t1, device)
    t1.eval()
    for p in t1.parameters():
        p.requires_grad_(False)
    print(f"t1 ({t1_view}): {args.t1_ckpt}  [FROZEN]")

    # ── t2 (fine-tuned branch) ────────────────────────────────────────────────
    t2 = _build_model(t2_def["in_channels"]).to(device)
    if args.t2_ckpt and Path(args.t2_ckpt).exists():
        load_checkpoint(args.t2_ckpt, t2, device)
        print(f"t2 ({t2_view}): {args.t2_ckpt}")
    else:
        print(f"t2 ({t2_view}): random init")

    # ── t3 (fine-tuned branch) ────────────────────────────────────────────────
    t3 = _build_model(t3_def["in_channels"]).to(device)
    if args.t3_ckpt and Path(args.t3_ckpt).exists():
        load_checkpoint(args.t3_ckpt, t3, device)
        print(f"t3 ({t3_view}): {args.t3_ckpt}")
    else:
        print(f"t3 ({t3_view}): random init")

    # ── student S (fine-tuned) ────────────────────────────────────────────────
    s_model = _build_model(STUDENT_IN_CH).to(device)
    if args.student_ckpt and Path(args.student_ckpt).exists():
        load_checkpoint(args.student_ckpt, s_model, device)
        print(f"S (1L50): {args.student_ckpt}")
    elif args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        load_encoder_init(s_model, args.student_init_ckpt, strict=False, device=device)
        print(f"S (1L50) SimCLR init: {args.student_init_ckpt}")
    else:
        print("S (1L50): random init")

    # ── data ──────────────────────────────────────────────────────────────────
    lead = args.lead or cfg["data"].get("lead", "II")
    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pf = 4 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0), prefetch_factor=pf)

    # ── optimizers & schedulers ───────────────────────────────────────────────
    bce_fn   = nn.BCEWithLogitsLoss()
    opt_t2   = torch.optim.AdamW(t2.parameters(),      lr=lr_, weight_decay=wd)
    opt_t3   = torch.optim.AdamW(t3.parameters(),      lr=lr_, weight_decay=wd)
    opt_s    = torch.optim.AdamW(s_model.parameters(), lr=lr_, weight_decay=wd)
    sched_t2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t2, T_max=ep)
    sched_t3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t3, T_max=ep)
    sched_s  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s,  T_max=ep)

    # ── training loop ─────────────────────────────────────────────────────────
    best_s_auc = best_t2_auc = best_t3_auc = 0.0
    s_log_rows = []; t2_log_rows = []; t3_log_rows = []

    t1_key = t1_def["input_key"]
    t2_key = t2_def["input_key"]
    t3_key = t3_def["input_key"]

    for epoch in range(1, ep + 1):
        t2.train(); t3.train(); s_model.train()
        ep_t2_bce = ep_t2_kd = ep_t2_sa = ep_t2_tot = 0.0
        ep_t3_bce = ep_t3_kd = ep_t3_sa = ep_t3_tot = 0.0
        ep_s_bce  = ep_s_kd  = ep_s_tot = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}", dynamic_ncols=True, leave=False):
            y    = batch["y"].to(device)
            s_in = batch[STUDENT_KEY].to(device)

            t2_out   = t2(batch[t2_key].to(device), return_features=False)
            t2_logits = t2_out["logits"]

            t3_out    = t3(batch[t3_key].to(device), return_features=False)
            t3_logits = t3_out["logits"]

            s_out     = s_model(s_in, return_features=False)
            s_logits  = s_out["logits"]

            with torch.no_grad():
                t1_out    = t1(batch[t1_key].to(device), return_features=False)
                t1_logits = t1_out["logits"]

            # L_t2 = BCE(t2) + lambda_kd*KD(t1→t2) + lambda_sa*KD(S.detach()→t2)
            loss_t2_bce = bce_fn(t2_logits, y)
            loss_t2_kd  = multi_label_kd_loss(t2_logits, t1_logits,          temperature)
            loss_t2_sa  = multi_label_kd_loss(t2_logits, s_logits.detach(),  temperature)
            loss_t2     = loss_t2_bce + lambda_kd * loss_t2_kd + lambda_sa * loss_t2_sa

            opt_t2.zero_grad(); loss_t2.backward();
            if gc > 0: nn.utils.clip_grad_norm_(t2.parameters(), gc)
            opt_t2.step()

            # L_t3 = BCE(t3) + lambda_kd*KD(t1→t3) + lambda_sa*KD(S.detach()→t3)
            loss_t3_bce = bce_fn(t3_logits, y)
            loss_t3_kd  = multi_label_kd_loss(t3_logits, t1_logits,          temperature)
            loss_t3_sa  = multi_label_kd_loss(t3_logits, s_logits.detach(),  temperature)
            loss_t3     = loss_t3_bce + lambda_kd * loss_t3_kd + lambda_sa * loss_t3_sa

            opt_t3.zero_grad(); loss_t3.backward()
            if gc > 0: nn.utils.clip_grad_norm_(t3.parameters(), gc)
            opt_t3.step()

            # L_S = BCE(S) + lambda_kd*[w1*KD(t1→S) + w2*KD(t2.detach()→S) + w3*KD(t3.detach()→S)]
            loss_s_bce  = bce_fn(s_logits, y)
            kd_s1       = multi_label_kd_loss(s_logits, t1_logits,           temperature)
            kd_s2       = multi_label_kd_loss(s_logits, t2_logits.detach(),  temperature)
            kd_s3       = multi_label_kd_loss(s_logits, t3_logits.detach(),  temperature)
            loss_s_kd   = w1 * kd_s1 + w2 * kd_s2 + w3 * kd_s3
            loss_s      = loss_s_bce + lambda_kd * loss_s_kd

            opt_s.zero_grad(); loss_s.backward()
            if gc > 0: nn.utils.clip_grad_norm_(s_model.parameters(), gc)
            opt_s.step()

            ep_t2_bce += loss_t2_bce.item(); ep_t2_kd += loss_t2_kd.item()
            ep_t2_sa  += loss_t2_sa.item();  ep_t2_tot += loss_t2.item()
            ep_t3_bce += loss_t3_bce.item(); ep_t3_kd += loss_t3_kd.item()
            ep_t3_sa  += loss_t3_sa.item();  ep_t3_tot += loss_t3.item()
            ep_s_bce  += loss_s_bce.item();  ep_s_kd   += loss_s_kd.item()
            ep_s_tot  += loss_s.item()
            n_batches += 1

        sched_t2.step(); sched_t3.step(); sched_s.step()

        s_probs,  s_labels,  _ = _run_eval(s_model, val_loader, STUDENT_KEY, device)
        t2_probs, t2_labels, _ = _run_eval(t2,      val_loader, t2_key,      device)
        t3_probs, t3_labels, _ = _run_eval(t3,      val_loader, t3_key,      device)
        s_val_auc  = compute_metrics(s_labels,  s_probs,  threshold=0.5)["macro_auc"]
        t2_val_auc = compute_metrics(t2_labels, t2_probs, threshold=0.5)["macro_auc"]
        t3_val_auc = compute_metrics(t3_labels, t3_probs, threshold=0.5)["macro_auc"]

        meta = {"sf_id": sf_id, "lambda_kd": lambda_kd, "lambda_sa": lambda_sa}
        if s_val_auc  > best_s_auc:
            best_s_auc  = s_val_auc
            save_checkpoint(s_ckpt,  s_model, epoch, {"val_auc": s_val_auc},  meta)
        if t2_val_auc > best_t2_auc:
            best_t2_auc = t2_val_auc
            save_checkpoint(t2_ckpt, t2,      epoch, {"val_auc": t2_val_auc}, meta)
        if t3_val_auc > best_t3_auc:
            best_t3_auc = t3_val_auc
            save_checkpoint(t3_ckpt, t3,      epoch, {"val_auc": t3_val_auc}, meta)

        n = n_batches
        print(f"Ep {epoch:3d} | "
              f"S: bce={ep_s_bce/n:.4f} kd={ep_s_kd/n:.4f} tot={ep_s_tot/n:.4f} "
              f"auc={s_val_auc:.4f}{'★' if s_val_auc==best_s_auc else ' '} | "
              f"t2: bce={ep_t2_bce/n:.4f} kd={ep_t2_kd/n:.4f} sa={ep_t2_sa/n:.4f} "
              f"auc={t2_val_auc:.4f}{'★' if t2_val_auc==best_t2_auc else ' '} | "
              f"t3: bce={ep_t3_bce/n:.4f} sa={ep_t3_sa/n:.4f} auc={t3_val_auc:.4f}"
              f"{'★' if t3_val_auc==best_t3_auc else ' '}")

        s_log_rows.append({"epoch": epoch,
                            "bce": round(ep_s_bce/n, 5), "kd": round(ep_s_kd/n, 5),
                            "total": round(ep_s_tot/n, 5),
                            "val_auc": round(s_val_auc, 5), "best_val_auc": round(best_s_auc, 5)})
        t2_log_rows.append({"epoch": epoch,
                             "bce": round(ep_t2_bce/n, 5), "kd": round(ep_t2_kd/n, 5),
                             "sa": round(ep_t2_sa/n, 5), "total": round(ep_t2_tot/n, 5),
                             "val_auc": round(t2_val_auc, 5), "best_val_auc": round(best_t2_auc, 5)})
        t3_log_rows.append({"epoch": epoch,
                             "bce": round(ep_t3_bce/n, 5), "kd": round(ep_t3_kd/n, 5),
                             "sa": round(ep_t3_sa/n, 5), "total": round(ep_t3_tot/n, 5),
                             "val_auc": round(t3_val_auc, 5), "best_val_auc": round(best_t3_auc, 5)})

    logs_dir = out_dir / "logs"
    for rows, name in [(s_log_rows, s_out_name), (t2_log_rows, t2_out_name), (t3_log_rows, t3_out_name)]:
        with open(logs_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

    # ── final evaluation ──────────────────────────────────────────────────────
    meta_full = {"seed": seed, "sf_id": sf_id, "t1": t1_view, "t2": t2_view, "t3": t3_view,
                 "weights": [w1, w2, w3], "lambda_kd": lambda_kd, "lambda_sa": lambda_sa,
                 "temperature": temperature, "method": "student_aware"}

    print(f"\nLoading best student checkpoint (val_auc={best_s_auc:.4f}) …")
    load_checkpoint(s_ckpt, s_model, device)
    s_metrics = _final_eval(s_model, val_loader, test_loader,
                            STUDENT_KEY, device, s_out_name, out_dir, meta_full)

    print(f"\nLoading best t2 checkpoint (val_auc={best_t2_auc:.4f}) …")
    load_checkpoint(t2_ckpt, t2, device)
    t2_metrics = _final_eval(t2, val_loader, test_loader,
                             t2_key, device, t2_out_name, out_dir, meta_full)

    print(f"\nLoading best t3 checkpoint (val_auc={best_t3_auc:.4f}) …")
    load_checkpoint(t3_ckpt, t3, device)
    t3_metrics = _final_eval(t3, val_loader, test_loader,
                             t3_key, device, t3_out_name, out_dir, meta_full)

    for label, m in [("Student (1L50)", s_metrics),
                     (f"t2 ({t2_view})", t2_metrics),
                     (f"t3 ({t3_view})", t3_metrics)]:
        auc = m["macro_auc"]; f1t = m["macro_f1_tuned"]
        print(f"\n{label}: AUC={auc:.4f}  F1_tuned={f1t:.4f}  "
              f"vs Prog={auc-PROG_AUC:+.4f}  vs MVKT={auc-MVKT_AUC:+.4f}  "
              f"Beats MVKT: {'YES ✓' if auc>MVKT_AUC else 'no'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",        required=True)
    p.add_argument("--lead",            default="II")
    p.add_argument("--sf_id",           default="SF02", choices=list(SF_STRUCTURES))
    p.add_argument("--t1_ckpt",         required=True,
                   help="Frozen t1 checkpoint (12L100 global teacher)")
    p.add_argument("--t2_ckpt",         default=None,
                   help="Pre-trained t2 branch checkpoint (from fork-join/skip-fork)")
    p.add_argument("--t3_ckpt",         default=None,
                   help="Pre-trained t3 branch checkpoint")
    p.add_argument("--student_ckpt",    default=None,
                   help="Pre-trained student checkpoint (from skip-fork)")
    p.add_argument("--student_init_ckpt", default=None,
                   help="SimCLR encoder for student init (used if --student_ckpt absent)")
    p.add_argument("--lambda_kd",       type=float, default=None)
    p.add_argument("--lambda_sa",       type=float, default=None)
    p.add_argument("--temperature",     type=float, default=None)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--output_base",     default=None)
    p.add_argument("--output_dir",      default=None)
    p.add_argument("--debug",           action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
