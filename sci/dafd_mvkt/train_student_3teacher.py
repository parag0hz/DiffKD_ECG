"""
Train 1-lead 50Hz student with 3-teacher multi-teacher KD.

Loss:
    L = BCE + lambda_kd * MultiTeacherKD(equal weights)

Usage:
    python dafd_mvkt/train_student_3teacher.py \\
        --config dafd_mvkt/configs/student_50hz_3teacher.yaml \\
        --data_dir comper_repo/ptb_xl \\
        --lead II \\
        --teacher_combo C06 \\
        --teacher_ckpts_json dafd_mvkt/configs/teacher_bank_II.json \\
        --teacher_weights 0.333,0.333,0.333 \\
        --lambda_kd 1.0 \\
        --temperature 2.0 \\
        --student_init_ckpt dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt \\
        --seed 0 \\
        --output_name student_II_50hz_3T_C06_equal_simclr_seed0
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

from data.ptbxl_multiteacher_dataset import PTBXLMultiTeacherDataset
from losses.multi_teacher_kd import MultiTeacherKDLoss
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import load_encoder_init


_HERE = Path(__file__).parent

# BCE 50Hz baseline references for reporting
_BCE_50HZ_AUC = 0.8061
_BCE_50HZ_F1  = 0.5740
_BEST_50HZ_AUC = 0.8396
_BEST_50HZ_F1  = 0.6176
_MVKT_AUC = 0.843
_MVKT_F1  = 0.626


# ── model builder ─────────────────────────────────────────────────────────────

def _build_resnet(in_channels: int, num_classes: int = 5,
                  base_channels: int = 64, proj_dim: int = 128) -> ResNet1d:
    return ResNet1d(
        in_channels   = in_channels,
        num_classes   = num_classes,
        layers        = [3, 4, 6, 3],
        base_channels = base_channels,
        proj_dim      = proj_dim,
        dropout       = 0.0,
    )


def _build_student(cfg: dict) -> ResNet1d:
    mc = cfg["model"]
    return ResNet1d(
        in_channels   = mc["in_channels"],
        num_classes   = mc["num_classes"],
        layers        = mc.get("layers", [3, 4, 6, 3]),
        base_channels = mc.get("base_channels", 64),
        proj_dim      = mc.get("proj_dim", 128),
        dropout       = mc.get("dropout", 0.0),
    )


# ── teacher loading ───────────────────────────────────────────────────────────

def load_teacher(teacher_cfg: dict, device: torch.device) -> tuple[ResNet1d, str]:
    """Load a single frozen teacher. Returns (model, input_key)."""
    in_ch   = teacher_cfg["in_channels"]
    ckpt    = teacher_cfg["checkpoint"]
    key     = teacher_cfg["input_key"]

    model = _build_resnet(in_ch)
    load_checkpoint(ckpt, model, device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, key


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    model: ResNet1d,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x   = batch["student_x"].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels, all_ids


# ── training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed if args.seed is not None else tc.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed: {seed}")

    lambda_kd   = args.lambda_kd   if args.lambda_kd   is not None else tc.get("lambda_kd",   1.0)
    temperature = args.temperature if args.temperature is not None else tc.get("temperature", 2.0)

    # ── teacher bank ──────────────────────────────────────────────────────────
    with open(args.teacher_ckpts_json) as f:
        bank = json.load(f)
    teacher_defs = bank["teachers"]

    # ── resolve combo ─────────────────────────────────────────────────────────
    with open(str(_HERE / "configs/teacher_combos_3of6.yaml")) as f:
        combos_cfg = yaml.safe_load(f)
    combo_def = combos_cfg["combos"][args.teacher_combo]
    teacher_ids = combo_def["teachers"]   # e.g. ["T1", "T3", "T5"]

    if args.teacher_weights:
        weights = [float(w) for w in args.teacher_weights.split(",")]
        assert len(weights) == 3, "--teacher_weights must have exactly 3 values"
    else:
        weights = combo_def["weights"]    # default from yaml

    print(f"Combo: {args.teacher_combo} → teachers: {teacher_ids}  weights: {weights}")

    # ── load teachers ─────────────────────────────────────────────────────────
    teachers, input_keys = [], []
    for tid in teacher_ids:
        tdef = teacher_defs[tid]
        model, key = load_teacher(tdef, device)
        teachers.append(model)
        input_keys.append(key)
        print(f"  Loaded {tid} ({tdef['label']}) from {tdef['checkpoint']}  key={key}")

    if args.debug:
        print(f"\n[DEBUG] Teacher input keys: {input_keys}")

    # ── dataset ───────────────────────────────────────────────────────────────
    lead = args.lead or cfg["data"].get("lead", "II")
    num_workers = tc.get("num_workers", 4)
    batch_size  = tc.get("batch_size", 32)

    print("Building datasets …")
    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── student ───────────────────────────────────────────────────────────────
    student = _build_student(cfg).to(device)
    if args.student_init_ckpt:
        load_encoder_init(student, args.student_init_ckpt, strict=False, device=device)
        print(f"Student initialized from: {args.student_init_ckpt}")
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {n_params/1e6:.2f}M")

    # ── debug: first batch shapes ─────────────────────────────────────────────
    if args.debug:
        first = next(iter(train_loader))
        print("\n[DEBUG] First batch:")
        print(f"  student_x shape: {first['student_x'].shape}")
        with torch.no_grad():
            s_out = student(first["student_x"].to(device), return_features=False)
        print(f"  student logits shape: {s_out['logits'].shape}")
        for tid, key in zip(teacher_ids, input_keys):
            x_t = first[key].to(device)
            print(f"  {tid} input ({key}) shape: {x_t.shape}")
            t_out = teachers[teacher_ids.index(tid)](x_t, return_features=False)
            print(f"  {tid} logits shape: {t_out['logits'].shape}")

    # ── losses ────────────────────────────────────────────────────────────────
    bce_fn = nn.BCEWithLogitsLoss()
    mkd_fn = MultiTeacherKDLoss(temperature=temperature)

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-4),
    )
    epochs = tc.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── output paths ──────────────────────────────────────────────────────────
    out_name = args.output_name or (
        f"student_II_50hz_3T_{args.teacher_combo}_equal"
        f"{'_simclr' if args.student_init_ckpt else ''}_seed{seed}"
    )
    out_dir  = Path(cfg.get("outputs", {}).get("dir", "dafd_mvkt/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"{out_name}_best.pt"
    log_path  = log_dir / f"{out_name}.csv"
    print(f"Output: {ckpt_path}")

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    log_rows  = []
    grad_clip = tc.get("grad_clip", 1.0)

    for epoch in range(1, epochs + 1):
        student.train()
        ep_bce = ep_kd = ep_total = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", dynamic_ncols=True,
                          leave=False):
            y = batch["y"].to(device)
            s_in = batch["student_x"].to(device)

            # student forward
            s_out = student(s_in, return_features=False)
            s_logits = s_out["logits"]   # [B, C]

            # teacher forwards (no_grad)
            t_logits_list = []
            with torch.no_grad():
                for teacher, key in zip(teachers, input_keys):
                    t_in  = batch[key].to(device)
                    t_out = teacher(t_in, return_features=False)
                    t_logits_list.append(t_out["logits"])

            # losses
            loss_bce = bce_fn(s_logits, y)
            loss_kd, _ = mkd_fn(s_logits, t_logits_list, weights)
            loss = loss_bce + lambda_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

            ep_bce   += loss_bce.item()
            ep_kd    += loss_kd.item()
            ep_total += loss.item()
            n_batches += 1

        scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        val_probs, val_labels, _ = run_eval(student, val_loader, device)
        val_m = compute_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = val_m["macro_auc"]

        avg_bce   = ep_bce   / n_batches
        avg_kd    = ep_kd    / n_batches
        avg_total = ep_total / n_batches

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, student, epoch,
                            {"val_auc": val_auc}, cfg,
                            extra={"combo": args.teacher_combo, "teachers": teacher_ids})

        print(f"Ep {epoch:3d} | bce={avg_bce:.4f} kd={avg_kd:.4f} total={avg_total:.4f} "
              f"| val_auc={val_auc:.4f} {'★' if is_best else ''}")

        log_rows.append({
            "epoch": epoch, "bce": round(avg_bce, 5), "kd": round(avg_kd, 5),
            "total": round(avg_total, 5), "val_auc": round(val_auc, 5),
            "best_val_auc": round(best_auc, 5),
        })

    # ── save training log ─────────────────────────────────────────────────────
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    # ── final evaluation with tuned thresholds ────────────────────────────────
    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(ckpt_path, student, device)

    # tune thresholds on val
    val_probs, val_labels, _ = run_eval(student, val_loader, device)
    grid = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    # evaluate on test
    test_probs, test_labels, test_ids = run_eval(student, test_loader, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    from data.ptbxl_multiteacher_dataset import SUPERCLASSES
    import pandas as pd

    print(f"\n{'='*62}")
    print(f"Combo: {args.teacher_combo}  Teachers: {teacher_ids}")
    print(f"{'='*62}")
    print(f"Macro AUC        : {m05['macro_auc']:.4f}")
    print(f"Macro F1  @0.5   : {m05.get('macro_f1', 0):.4f}")
    print(f"Macro F1  tuned  : {m_tune.get('macro_f1', 0):.4f}")
    print(f"Beats BCE 50Hz   : {'YES' if m05['macro_auc'] > _BCE_50HZ_AUC else 'no'}")
    print(f"Beats current best: {'YES' if m05['macro_auc'] > _BEST_50HZ_AUC else 'no'}")
    print(f"Beats MVKT AUC   : {'YES' if m05['macro_auc'] > _MVKT_AUC else 'no'}")

    # build metrics dict
    metrics = {
        "combo": args.teacher_combo,
        "teachers": teacher_ids,
        "weights": weights,
        "macro_auc":          m05["macro_auc"],
        "macro_f1_0_5":       m05.get("macro_f1", 0),
        "macro_f1_tuned":     m_tune.get("macro_f1", 0),
        "macro_precision_0_5": m05.get("macro_precision", 0),
        "macro_recall_0_5":    m05.get("macro_recall", 0),
        "macro_specificity_0_5": m05.get("macro_specificity", 0),
        "per_class_thresholds": {
            SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)
        },
        "_meta": {
            "output_name": out_name,
            "seed": seed,
            "lambda_kd": lambda_kd,
            "temperature": temperature,
            "student_init": str(args.student_init_ckpt) if args.student_init_ckpt else None,
        },
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"] = m05.get(f"auc_{c}", float("nan"))
        metrics[f"f1_0_5_{c}"]   = m05.get(f"f1_{c}",  float("nan"))
        metrics[f"f1_tuned_{c}"] = m_tune.get(f"f1_{c}", float("nan"))

    metrics_path = out_dir / f"metrics_test_{out_name}.json"
    with open(metrics_path, "w") as f:
        json.dump({k: (round(v, 5) if isinstance(v, float) else v)
                   for k, v in metrics.items()}, f, indent=2)

    # predictions CSV
    preds = (test_probs >= thresholds[None]).astype(int)
    rows  = []
    for i, rid in enumerate(test_ids):
        row = {"record_id": rid}
        for ci, c in enumerate(SUPERCLASSES):
            row[f"y_{c}"]    = int(test_labels[i, ci])
            row[f"prob_{c}"] = round(float(test_probs[i, ci]), 5)
            row[f"pred_{c}"] = int(preds[i, ci])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"predictions_test_{out_name}.csv", index=False)

    print(f"\nSaved: {metrics_path}")
    print(f"       {out_dir}/predictions_test_{out_name}.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",             required=True)
    p.add_argument("--data_dir",           required=True)
    p.add_argument("--lead",               default="II")
    p.add_argument("--teacher_combo",      required=True,
                   help="Combo ID e.g. C06")
    p.add_argument("--teacher_ckpts_json", required=True,
                   help="Path to teacher_bank_II.json")
    p.add_argument("--teacher_weights",    default=None,
                   help="Comma-separated weights, e.g. 0.333,0.333,0.333")
    p.add_argument("--lambda_kd",          type=float, default=None)
    p.add_argument("--temperature",        type=float, default=None)
    p.add_argument("--student_init_ckpt",  default=None,
                   help="SimCLR encoder checkpoint for student init")
    p.add_argument("--seed",               type=int, default=None)
    p.add_argument("--output_name",        default=None)
    p.add_argument("--debug",              action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
