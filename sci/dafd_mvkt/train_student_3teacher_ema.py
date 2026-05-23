"""
3-teacher KD with EMA teacher regularization for 1-lead 50Hz student.

Loss:
    L = BCE
      + lambda_kd  * MultiTeacherKD(T1, T2, T3 → S)
      + lambda_ema * KD(EMA(S) → S)

EMA copy of the student provides consistency regularization.
EMA is updated after each optimizer.step().

Usage:
    python dafd_mvkt/train_student_3teacher_ema.py \\
        --config           dafd_mvkt/configs/adaptive_50hz.yaml \\
        --data_dir         comper_repo/ptb_xl \\
        --lead             II \\
        --teacher_combo    C14 \\
        --teacher_ckpts_json dafd_mvkt/configs/teacher_bank_II.json \\
        --lambda_ema       0.1 \\
        --ema_decay        0.999 \\
        --seed             0 \\
        --output_name      student_II_50hz_3T_C14_ema_lam0.1_decay0.999_seed0
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
from losses.multi_teacher_kd import MultiTeacherKDLoss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ema import ModelEMA
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import load_encoder_init

BCE_AUC = 0.8061; BCE_F1  = 0.5740
PROG_AUC = 0.8396; PROG_F1 = 0.6176
C14_AUC  = 0.8425; C14_F1  = 0.6243
MVKT_AUC = 0.843;  MVKT_F1 = 0.626


def _build_resnet(in_channels: int, num_classes: int = 5) -> ResNet1d:
    return ResNet1d(in_channels=in_channels, num_classes=num_classes,
                   layers=[3, 4, 6, 3], base_channels=64,
                   proj_dim=128, dropout=0.0)


def _build_student(cfg: dict) -> ResNet1d:
    mc = cfg["model"]
    return ResNet1d(in_channels=mc["in_channels"], num_classes=mc["num_classes"],
                   layers=mc.get("layers", [3, 4, 6, 3]),
                   base_channels=mc.get("base_channels", 64),
                   proj_dim=mc.get("proj_dim", 128),
                   dropout=mc.get("dropout", 0.0))


@torch.no_grad()
def _run_eval(model, loader, device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x = batch["student_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels, all_ids


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

    lambda_kd   = args.lambda_kd   if args.lambda_kd   is not None else tc.get("lambda_kd",   1.0)
    lambda_ema  = args.lambda_ema  if args.lambda_ema  is not None else tc.get("lambda_ema",   0.1)
    ema_decay   = args.ema_decay   if args.ema_decay   is not None else tc.get("ema_decay",   0.999)
    temperature = args.temperature if args.temperature is not None else tc.get("temperature",  2.0)

    # ── teacher bank ──────────────────────────────────────────────────────────
    with open(args.teacher_ckpts_json) as f:
        bank = json.load(f)
    teacher_defs = bank["teachers"]

    with open(str(_HERE / "configs/teacher_combos_3of6.yaml")) as f:
        combos_cfg = yaml.safe_load(f)
    combo_def   = combos_cfg["combos"][args.teacher_combo]
    teacher_ids = combo_def["teachers"]
    weights     = ([float(w) for w in args.teacher_weights.split(",")]
                   if args.teacher_weights else combo_def["weights"])
    assert len(weights) == 3

    print(f"\n{'='*60}")
    print(f"3T-EMA  combo={args.teacher_combo}  teachers={teacher_ids}")
    print(f"lambda_kd={lambda_kd}  lambda_ema={lambda_ema}  decay={ema_decay}  T={temperature}")
    print(f"device={device}  seed={seed}")
    print(f"{'='*60}")

    teachers, input_keys = [], []
    for tid in teacher_ids:
        td = teacher_defs[tid]
        m  = _build_resnet(td["in_channels"])
        load_checkpoint(td["checkpoint"], m, device)
        m.to(device).eval()
        for p in m.parameters(): p.requires_grad_(False)
        teachers.append(m)
        input_keys.append(td["input_key"])
        print(f"  Loaded {tid} ({td['label']}) key={td['input_key']}")

    # ── dataset ───────────────────────────────────────────────────────────────
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

    # ── student + EMA ─────────────────────────────────────────────────────────
    student = _build_student(cfg).to(device)
    if args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        load_encoder_init(student, args.student_init_ckpt, strict=False, device=device)
        print(f"Student initialized from: {args.student_init_ckpt}")
    else:
        print("Student: random init")
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {n_params/1e6:.2f}M")

    ema = ModelEMA(student, decay=ema_decay)
    ema.shadow.to(device)
    print(f"EMA model created  decay={ema_decay}")

    # ── losses & optimizer ────────────────────────────────────────────────────
    bce_fn = nn.BCEWithLogitsLoss()
    mkd_fn = MultiTeacherKDLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    # ── output paths ──────────────────────────────────────────────────────────
    out_name = args.output_name or (
        f"student_II_50hz_3T_{args.teacher_combo}_ema"
        f"_lam{lambda_ema}_decay{str(ema_decay).replace('.','')}_seed{seed}")
    out_dir  = Path(cfg.get("outputs", {}).get("dir", "dafd_mvkt/outputs/adaptive"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    ckpt_path = out_dir / f"{out_name}_best.pt"
    log_path  = out_dir / "logs" / f"{out_name}.csv"
    print(f"Output: {ckpt_path}")

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    log_rows  = []

    for epoch in range(1, ep + 1):
        student.train()
        ep_bce = ep_kd = ep_ema = ep_tot = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}", dynamic_ncols=True, leave=False):
            y    = batch["y"].to(device)
            s_in = batch["student_x"].to(device)

            s_out    = student(s_in, return_features=False)
            s_logits = s_out["logits"]

            # multi-teacher KD (teachers frozen)
            t_logits_list = []
            with torch.no_grad():
                for teacher, key in zip(teachers, input_keys):
                    t_out = teacher(batch[key].to(device), return_features=False)
                    t_logits_list.append(t_out["logits"])

            # EMA KD (EMA model frozen)
            with torch.no_grad():
                ema_out    = ema(s_in, return_features=False)
                ema_logits = ema_out["logits"]

            loss_bce, _ = bce_fn(s_logits, y), None
            loss_bce = bce_fn(s_logits, y)
            loss_kd, _ = mkd_fn(s_logits, t_logits_list, weights)
            loss_ema = multi_label_kd_loss(s_logits, ema_logits, temperature)
            loss     = loss_bce + lambda_kd * loss_kd + lambda_ema * loss_ema

            optimizer.zero_grad()
            loss.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(student.parameters(), gc)
            optimizer.step()
            ema.update(student)

            ep_bce  += loss_bce.item()
            ep_kd   += loss_kd.item()
            ep_ema  += loss_ema.item()
            ep_tot  += loss.item()
            n_batches += 1

        scheduler.step()

        val_probs, val_labels, _ = _run_eval(student, val_loader, device)
        val_auc = compute_metrics(val_labels, val_probs, threshold=0.5)["macro_auc"]

        avg_bce = ep_bce / n_batches
        avg_kd  = ep_kd  / n_batches
        avg_ema = ep_ema / n_batches
        avg_tot = ep_tot / n_batches
        is_best = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, student, epoch, {"val_auc": val_auc},
                            {"combo": args.teacher_combo, "lambda_ema": lambda_ema,
                             "ema_decay": ema_decay})

        print(f"Ep {epoch:3d} | bce={avg_bce:.4f} kd={avg_kd:.4f} ema={avg_ema:.4f} "
              f"tot={avg_tot:.4f} | val_auc={val_auc:.4f} {'★' if is_best else ''}")

        log_rows.append({"epoch": epoch, "bce": round(avg_bce, 5), "kd": round(avg_kd, 5),
                         "kd_ema": round(avg_ema, 5), "total": round(avg_tot, 5),
                         "val_auc": round(val_auc, 5), "best_val_auc": round(best_auc, 5)})

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    # ── final evaluation ──────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(ckpt_path, student, device)

    val_probs, val_labels, _     = _run_eval(student, val_loader,  device)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(student, test_loader, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    metrics = {
        "output_name": out_name, "combo": args.teacher_combo,
        "lambda_ema": lambda_ema, "ema_decay": ema_decay,
        "macro_auc":       round(m05["macro_auc"], 5),
        "macro_f1_0_5":    round(m05.get("macro_f1", 0), 5),
        "macro_f1_tuned":  round(m_tune.get("macro_f1", 0), 5),
        "val_auc_best":    round(best_auc, 5),
        "per_class_thresholds": {SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)},
        "_meta": {"seed": seed, "lambda_kd": lambda_kd, "lambda_ema": lambda_ema,
                  "ema_decay": ema_decay, "temperature": temperature},
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    with open(out_dir / f"metrics_val_{out_name}.json", "w") as f:
        json.dump({"output_name": out_name,
                   "macro_auc": round(val_m05["macro_auc"], 5),
                   "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5)}, f, indent=2)
    with open(out_dir / f"metrics_test_{out_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows  = [{"record_id": rid,
              **{f"y_{c}": int(test_labels[i, ci]) for ci, c in enumerate(SUPERCLASSES)},
              **{f"prob_{c}": round(float(test_probs[i, ci]), 5) for ci, c in enumerate(SUPERCLASSES)},
              **{f"pred_{c}": int(preds[i, ci]) for ci, c in enumerate(SUPERCLASSES)}}
             for i, rid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(out_dir / f"predictions_test_{out_name}.csv", index=False)

    auc = metrics["macro_auc"]; f1t = metrics["macro_f1_tuned"]
    print(f"\nAUC={auc:.4f}  F1_tuned={f1t:.4f}")
    print(f"vs Prog : {auc-PROG_AUC:+.4f}  vs C14: {auc-C14_AUC:+.4f}  vs MVKT: {auc-MVKT_AUC:+.4f}")
    print(f"Beats MVKT AUC: {'YES ✓' if auc>MVKT_AUC else 'no'}  "
          f"F1: {'YES ✓' if f1t>MVKT_F1 else 'no'}")
    print(f"Saved: {out_dir}/metrics_test_{out_name}.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",             default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",           required=True)
    p.add_argument("--lead",               default="II")
    p.add_argument("--teacher_combo",      default="C14")
    p.add_argument("--teacher_ckpts_json", default="dafd_mvkt/configs/teacher_bank_II.json")
    p.add_argument("--teacher_weights",    default=None)
    p.add_argument("--student_init_ckpt",  default=None)
    p.add_argument("--lambda_kd",          type=float, default=None)
    p.add_argument("--lambda_ema",         type=float, default=None)
    p.add_argument("--ema_decay",          type=float, default=None)
    p.add_argument("--temperature",        type=float, default=None)
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--epochs",             type=int,   default=None)
    p.add_argument("--batch_size",         type=int,   default=None)
    p.add_argument("--output_name",        default=None)
    p.add_argument("--output_dir",         default=None)
    p.add_argument("--debug",              action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output_dir:
        import yaml as _yaml
        with open(args.config) as f:
            _c = _yaml.safe_load(f)
        _c.setdefault("outputs", {})["dir"] = args.output_dir
        args._cfg_override = _c
    train(args)
