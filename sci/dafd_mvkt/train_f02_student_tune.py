"""
F02 Student Optimization Trainer  (+ TG-MTA attention extension).

Trains the final 1L50 student using frozen F02 branch teachers
(1L500 and 1L100) with configurable:
  - teacher weight ratio (Phase 2)
  - CRF / FeatureKD from 1L100 branch (Phase 3)
  - temperature (Phase 4)
  - confidence-weighted KD (Phase 6)
  - TG-MTA / SE / CBAM attention + teacher-guided attention KD (Phase TG)

Loss:
    L = BCE
      + lambda_kd * (w500 * MKD(1L500→S) + w100 * MKD(1L100→S))
      + beta_crf     * CRF(1L100, S)              [if --use_crf]
      + gamma_feature * FeatureKD(1L100→S)        [if --use_feature_kd]
      + lambda_att   * AttentionKD(S_att, T_sal)  [if --lambda_att > 0]

Backward compatibility:
  New attention args all default to their "off" values so the existing
  F02W04 command produces exactly the same result as before.

Usage (F02W04 baseline, unchanged):
    python dafd_mvkt/train_f02_student_tune.py \\
        --config dafd_mvkt/configs/adaptive_50hz.yaml \\
        --data_dir comper_repo/ptb_xl --lead II \\
        --teacher_1l500_ckpt dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt \\
        --teacher_1l100_ckpt dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt \\
        --student_init_ckpt  dafd_mvkt/outputs/simclr/simclr_II_50hz_seed0_encoder.pt \\
        --weights 0.60,0.40 --temperature 2.0 \\
        --variant_id F02W04 --output_dir dafd_mvkt/outputs/f02_opt

Usage (TG-MTA + attention KD):
    python dafd_mvkt/train_f02_student_tune.py \\
        ... (same as above) ...
        --attention_type tgmta \\
        --attention_layers layer3,layer4 \\
        --lambda_att 0.01 \\
        --variant_id ATTN05 \\
        --output_dir dafd_mvkt/outputs/tgmta_f02
"""
from __future__ import annotations
import argparse
import csv
import json
import os
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
from losses.attention_kd import (
    make_dual_teacher_saliency,
    attention_kd_loss,
)
from losses.contrastive import cross_rate_contrastive_loss
from losses.feature_kd import FeatureKDLoss
from losses.mkd import multi_label_kd_loss, multi_label_kd_loss_per_class
from models.resnet1d import ResNet1d
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics, find_best_thresholds
from utils.seed import set_seed
from utils.simclr_checkpoint import load_encoder_init

F02_AUC  = 0.8447;  F02_F1  = 0.6275
C14_AUC  = 0.8425;  C14_F1  = 0.6243
PROG_AUC = 0.8396;  PROG_F1 = 0.6176
MVKT_AUC = 0.843;   MVKT_F1 = 0.626

T1L500_KEY = "teacher_1l_500"   # [B, 1, 5000]
T1L100_KEY = "teacher_1l_100"   # [B, 1, 1000]
STUDENT_KEY = "student_x"       # [B, 1, 500]

_AKD_MODES = [
    "base", "confidence", "agreement", "confidence_agreement",
    "ensemble_prob", "ensemble_logit",
    "class_reliability", "label_correlation", "dynamic_weight",
]

_RKDLR_MODES = ["gate_full", "degraded_target", "mixed_target"]

T12L100_KEY = "teacher_12l_100"   # [B, 12, 1000]

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


# ─────────────────────────────────────────────────────────────────────────────
# CLRD helpers
# ─────────────────────────────────────────────────────────────────────────────

class LeadImportanceHead(nn.Module):
    """Linear head: pooled [B,512] → lead_attention [B,5,12]."""
    def __init__(self, feat_dim: int = 512, num_classes: int = 5, num_leads: int = 12):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes * num_leads)
        self.C = num_classes
        self.L = num_leads

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, feat_dim]
        out = self.fc(z).view(z.shape[0], self.C, self.L)
        return _F.softmax(out, dim=-1)   # [B,5,12]


@torch.no_grad()
def _build_lead_importance_cache(
    teacher12: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.5,
    eps: float = 1e-8,
) -> dict[int, torch.Tensor]:
    """Pre-compute per-sample [5,12] lead importance tensors.

    importance[b,c,l] = ReLU(p_full[b,c] - p_occlude_l[b,c])
    Normalized to sum-1 over lead dimension.
    Returns dict: record_id(int) -> [5,12] float32 CPU tensor.
    """
    teacher12.eval()
    cache: dict[int, torch.Tensor] = {}
    num_leads = 12

    for batch in tqdm(loader, desc="Building CLRD cache", leave=False,
                      dynamic_ncols=True):
        x12  = batch[T12L100_KEY].to(device)   # [B,12,1000]
        rids = batch["record_id"].tolist()
        B    = x12.shape[0]

        p_full = torch.sigmoid(teacher12(x12)["logits"])   # [B,5]

        importance = torch.zeros(B, 5, num_leads, device=device)
        for l in range(num_leads):
            x_mask = x12.clone()
            x_mask[:, l, :] = 0.0
            p_mask = torch.sigmoid(teacher12(x_mask)["logits"])   # [B,5]
            importance[:, :, l] = (p_full - p_mask).clamp(min=0.0)

        # normalize over lead dimension; zero-sum → uniform
        s = importance.sum(dim=-1, keepdim=True)          # [B,5,1]
        uniform = torch.full_like(importance, 1.0 / num_leads)
        A = torch.where(s > eps, importance / (s + eps), uniform)  # [B,5,12]

        for i, rid in enumerate(rids):
            cache[int(rid)] = A[i].cpu()

    return cache


def _compute_clrd_loss(
    A_student: torch.Tensor,   # [B,5,12]  softmax output
    A_teacher: torch.Tensor,   # [B,5,12]  normalized importance (detached)
    y: torch.Tensor,           # [B,5]     ground-truth labels
    clrd_loss: str = "kl",
    positive_only: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL or MSE between student virtual lead attention and teacher importance."""
    B, C, L = A_student.shape

    if positive_only:
        mask = y.bool()                       # [B,5]
        if mask.sum() == 0:
            return A_student.new_zeros(1)
        # gather positive (b,c) pairs
        As = A_student[mask]                  # [N, 12]
        At = A_teacher[mask]                  # [N, 12]
    else:
        As = A_student.view(B * C, L)
        At = A_teacher.view(B * C, L)

    if clrd_loss == "kl":
        # KL(At || As): At * (log At - log As)
        log_As = (As + eps).log()
        log_At = (At + eps).log()
        loss   = (At * (log_At - log_As)).sum(dim=-1).mean()
    else:  # mse
        loss = _F.mse_loss(As, At)

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# RKDLR helpers
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn.functional as _F


def _degrade_signal(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample x to target_len then upsample back to original length."""
    orig_len = x.shape[-1]
    x_down = _F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return _F.interpolate(x_down, size=orig_len, mode="linear", align_corners=False)


def _compute_rkdlr_loss(
    s_logits: torch.Tensor,   # [B, C]
    l500_full: torch.Tensor,  # [B, C]  teacher500 full logits (detached)
    l100_full: torch.Tensor,  # [B, C]  teacher100 full logits (detached)
    l500_deg: torch.Tensor,   # [B, C]  teacher500 degraded logits (detached)
    l100_deg: torch.Tensor,   # [B, C]  teacher100 degraded logits (detached)
    w500: float,
    w100: float,
    temperature: float,
    rkdlr_mode: str,
    rkdlr_gamma: float = 1.0,
    rkdlr_min_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Recoverability-weighted KD loss.

    r[b,c] = probability that class c teacher signal survives 50Hz degradation.
    High r → teacher knows this, student can learn it → full weight.
    Low r  → teacher loses this at 50Hz → down-weight KD.
    """
    T = temperature
    _EPS = 1e-8
    info: dict = {}

    with torch.no_grad():
        p500_f = torch.sigmoid(l500_full / T)   # [B, C]
        p500_d = torch.sigmoid(l500_deg  / T)
        p100_f = torch.sigmoid(l100_full / T)
        p100_d = torch.sigmoid(l100_deg  / T)

        r500 = (1.0 - (p500_f - p500_d).abs()).clamp(0.0, 1.0)  # [B, C]
        r100 = (1.0 - (p100_f - p100_d).abs()).clamp(0.0, 1.0)
        r    = (w500 * r500 + w100 * r100).clamp(0.0, 1.0)

        if rkdlr_gamma != 1.0:
            r = r ** rkdlr_gamma
        if rkdlr_min_weight > 0.0:
            r = rkdlr_min_weight + (1.0 - rkdlr_min_weight) * r

        r = r.detach()

    info["mean_r500"]           = r500.mean().item()
    info["mean_r100"]           = r100.mean().item()
    info["mean_recoverability"] = r.mean().item()

    if rkdlr_mode == "gate_full":
        kd500 = multi_label_kd_loss_per_class(s_logits, l500_full, T)  # [B,C]
        kd100 = multi_label_kd_loss_per_class(s_logits, l100_full, T)
        loss  = (r * (w500 * kd500 + w100 * kd100)).mean()

    elif rkdlr_mode == "degraded_target":
        kd500 = multi_label_kd_loss_per_class(s_logits, l500_deg, T)
        kd100 = multi_label_kd_loss_per_class(s_logits, l100_deg, T)
        loss  = (w500 * kd500 + w100 * kd100).mean()

    elif rkdlr_mode == "mixed_target":
        with torch.no_grad():
            p500_mix = r * p500_f + (1.0 - r) * p500_d
            p100_mix = r * p100_f + (1.0 - r) * p100_d
            p_mix    = (w500 * p500_mix + w100 * p100_mix).clamp(_EPS, 1.0 - _EPS)
        loss = _F.binary_cross_entropy_with_logits(
            s_logits / T, p_mix, reduction="mean") * (T ** 2)

    else:
        raise ValueError(f"Unknown rkdlr_mode: {rkdlr_mode}")

    return loss, info


# ─────────────────────────────────────────────────────────────────────────────
# adaptive KD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_adaptive_kd_loss(
    s_logits: torch.Tensor,
    l500: torch.Tensor,
    l100: torch.Tensor,
    w500: float,
    w100: float,
    temperature: float,
    akd_mode: str,
    akd_gamma: float,
    akd_min_weight: float,
    akd_detach_weight: bool,
    akd_normalize_weight: bool,
    class_reliability: torch.Tensor | None,
    label_corr_weight: float,
    current_epoch: int,
    total_epochs: int,
    dynamic_w500_start: float,
    dynamic_w500_end: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns (kd_loss scalar, corr_loss scalar, info_dict).
    info_dict contains mean_akd_weight and current_w500/w100 for logging.
    """
    _EPS = 1e-8
    info: dict = {}

    # ── dynamic weight schedule ───────────────────────────────────────────────
    if akd_mode == "dynamic_weight":
        frac = (current_epoch - 1) / max(total_epochs - 1, 1)
        w500_cur = dynamic_w500_start + frac * (dynamic_w500_end - dynamic_w500_start)
        w100_cur = 1.0 - w500_cur
        info["current_w500"] = w500_cur
        info["current_w100"] = w100_cur
    else:
        w500_cur, w100_cur = w500, w100

    # ── ensemble targets ──────────────────────────────────────────────────────
    if akd_mode in ("ensemble_prob", "ensemble_logit"):
        T = temperature
        if akd_mode == "ensemble_prob":
            p_ens = (w500_cur * torch.sigmoid(l500.detach() / T)
                     + w100_cur * torch.sigmoid(l100.detach() / T))
        else:
            logit_ens = (w500_cur * l500.detach() + w100_cur * l100.detach())
            p_ens = torch.sigmoid(logit_ens / T)

        # BCE with soft target, T² scaling
        kd_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                s_logits / T, p_ens.detach(), reduction="mean"
            ) * (T ** 2)
        )
        corr_loss = s_logits.new_zeros(1)
        return kd_loss, corr_loss, info

    # ── base / dynamic_weight: use original KL-based loss (matches F02T01) ────
    if akd_mode in ("base", "dynamic_weight"):
        kd500_s = multi_label_kd_loss(s_logits, l500, temperature)
        kd100_s = multi_label_kd_loss(s_logits, l100, temperature)
        kd_loss = w500_cur * kd500_s + w100_cur * kd100_s
        corr_loss = s_logits.new_zeros(1)
        return kd_loss, corr_loss, info

    # ── per-class KD losses (for adaptive weighting modes) ───────────────────
    kd500 = multi_label_kd_loss_per_class(s_logits, l500, temperature)  # [B,C]
    kd100 = multi_label_kd_loss_per_class(s_logits, l100, temperature)  # [B,C]
    base_kd = w500_cur * kd500 + w100_cur * kd100                       # [B,C]

    # ── adaptive sample/class weights ────────────────────────────────────────
    if akd_mode in ("confidence", "agreement", "confidence_agreement"):
        T = temperature
        with torch.no_grad() if akd_detach_weight else torch.enable_grad():
            p500 = torch.sigmoid(l500 / T)   # [B,C]
            p100 = torch.sigmoid(l100 / T)   # [B,C]

            conf500 = 2.0 * (p500 - 0.5).abs()   # [B,C]  in [0,1]
            conf100 = 2.0 * (p100 - 0.5).abs()   # [B,C]  in [0,1]
            conf = w500_cur * conf500 + w100_cur * conf100   # [B,C]

            agreement = 1.0 - (p500 - p100).abs()           # [B,C]  in [0,1]

            if akd_mode == "confidence":
                q = conf
            elif akd_mode == "agreement":
                q = agreement
            else:  # confidence_agreement
                q = agreement * conf                          # [B,C]

        if akd_detach_weight:
            q = q.detach()

        q = q.clamp(0.0, 1.0) ** akd_gamma

        if akd_min_weight > 0.0:
            q = akd_min_weight + (1.0 - akd_min_weight) * q

        if akd_normalize_weight:
            q = q / (q.mean() + _EPS)

        info["mean_akd_weight"] = q.mean().item()
        kd_loss = (q * base_kd).mean()

    elif akd_mode == "class_reliability":
        # class_reliability: [C] tensor
        if class_reliability is not None:
            r = class_reliability.to(s_logits.device).unsqueeze(0)  # [1,C]
        else:
            r = s_logits.new_ones(1, s_logits.shape[1])
        kd_loss = (r * base_kd).mean()
        info["mean_akd_weight"] = r.mean().item()

    else:
        # base or dynamic_weight: plain weighted sum
        kd_loss = base_kd.mean()

    # ── label correlation loss ────────────────────────────────────────────────
    corr_loss = s_logits.new_zeros(1)
    if akd_mode == "label_correlation" and label_corr_weight > 0.0:
        T = temperature
        with torch.no_grad():
            p_t = (w500_cur * torch.sigmoid(l500 / T)
                   + w100_cur * torch.sigmoid(l100 / T))  # [B,C]
        p_s = torch.sigmoid(s_logits)                     # [B,C]
        B = p_s.shape[0]
        if B > 1:
            pt_c = p_t - p_t.mean(dim=0, keepdim=True)
            ps_c = p_s - p_s.mean(dim=0, keepdim=True)
            C_t = (pt_c.T @ pt_c) / (B - 1)              # [C,C]
            C_s = (ps_c.T @ ps_c) / (B - 1)              # [C,C]
            corr_loss = torch.nn.functional.mse_loss(C_s, C_t.detach())

    return kd_loss, corr_loss, info


# ─────────────────────────────────────────────────────────────────────────────
# model builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_model(
    in_channels: int,
    attention_type: str = "none",
    attention_layers: list[str] | None = None,
    use_wide_kernel: bool = False,
) -> ResNet1d:
    return ResNet1d(
        in_channels=in_channels, num_classes=5,
        layers=[3, 4, 6, 3], base_channels=64,
        proj_dim=128, dropout=0.0,
        attention_type=attention_type,
        attention_layers=attention_layers,
        use_wide_kernel=use_wide_kernel,
    )


# ─────────────────────────────────────────────────────────────────────────────
# evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _run_eval(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x   = batch[STUDENT_KEY].to(device)
        out = model(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
    return probs, torch.cat(all_labels).numpy(), all_ids


# ─────────────────────────────────────────────────────────────────────────────
# summary CSV helper
# ─────────────────────────────────────────────────────────────────────────────
def _update_summary(out_dir: Path, row: dict) -> None:
    """Append or update a row in summary.csv keyed by variant_id."""
    csv_path = out_dir / "summary.csv"
    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("variant_id") != row["variant_id"]]
    rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# attention map visualization
# ─────────────────────────────────────────────────────────────────────────────
def _save_attention_maps(
    student: nn.Module,
    t500: nn.Module,
    t100: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    variant_id: str,
    out_dir: Path,
    vis_n: int = 5,
    att_layer_names: list[str] | None = None,
    attn_w500: float = 0.60,
    attn_w100: float = 0.40,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping attention map saving")
        return

    save_dir = out_dir / "attention_maps" / variant_id
    save_dir.mkdir(parents=True, exist_ok=True)
    if att_layer_names is None:
        att_layer_names = ["layer4"]

    # collect samples per class
    class_samples: dict[str, list] = {c: [] for c in SUPERCLASSES}
    student.eval()

    with torch.no_grad():
        for batch in test_loader:
            xs    = batch[STUDENT_KEY].to(device)
            x500  = batch[T1L500_KEY].to(device)
            x100  = batch[T1L100_KEY].to(device)
            labels = batch["y"].cpu().numpy()
            rids   = batch["record_id"]

            s_out  = student(xs, return_features=True, return_attention=True)
            t5_out = t500(x500, return_features=True)
            t1_out = t100(x100, return_features=True)

            probs = torch.sigmoid(s_out["logits"]).cpu().numpy()

            for i in range(xs.shape[0]):
                for ci, c in enumerate(SUPERCLASSES):
                    if labels[i, ci] == 1 and len(class_samples[c]) < vis_n:
                        # compute teacher saliency for first attention layer
                        lay = att_layer_names[-1]
                        student_att = s_out.get("attentions", {}).get(lay)
                        L_s = student_att.shape[-1] if student_att is not None else 16
                        t_sal = make_dual_teacher_saliency(
                            t5_out["feature_map"][i:i+1],
                            t1_out["feature_map"][i:i+1],
                            target_len=L_s,
                            w500=attn_w500,
                            w100=attn_w100,
                        )
                        class_samples[c].append({
                            "waveform": xs[i, 0].cpu().numpy(),
                            "student_att": student_att[i, 0].cpu().numpy()
                                           if student_att is not None
                                           else np.ones(L_s),
                            "teacher_sal": t_sal[0, 0].cpu().numpy(),
                            "probs": probs[i],
                            "labels": labels[i],
                            "rid": rids[i],
                        })

            if all(len(v) >= vis_n for v in class_samples.values()):
                break

    # plot
    for c, samples in class_samples.items():
        for si, s in enumerate(samples):
            fig, axes = plt.subplots(3, 1, figsize=(12, 6))
            axes[0].plot(s["waveform"], lw=0.8, color="steelblue")
            axes[0].set_title(
                f"{c} — rid={s['rid']}  "
                f"P(NORM)={s['probs'][0]:.2f} P(MI)={s['probs'][1]:.2f} "
                f"P(STTC)={s['probs'][2]:.2f} P(CD)={s['probs'][3]:.2f} "
                f"P(HYP)={s['probs'][4]:.2f}",
                fontsize=8,
            )
            axes[0].set_ylabel("ECG (Lead-II)")

            t_ax = np.linspace(0, len(s["waveform"]) - 1, len(s["student_att"]))
            axes[1].fill_between(t_ax, s["student_att"], alpha=0.7,
                                  color="tomato")
            axes[1].set_ylim(0, 1.05)
            axes[1].set_ylabel("Student Att")

            t_ax_t = np.linspace(0, len(s["waveform"]) - 1, len(s["teacher_sal"]))
            axes[2].fill_between(t_ax_t, s["teacher_sal"], alpha=0.7,
                                  color="mediumseagreen")
            axes[2].set_ylim(0, 1.05)
            axes[2].set_ylabel("Teacher Sal")
            axes[2].set_xlabel("Time-step (50 Hz)")

            for ax in axes:
                ax.set_xlim(0, len(s["waveform"]) - 1)

            plt.tight_layout()
            plt.savefig(save_dir / f"{c}_{si:02d}.png", dpi=100)
            plt.close(fig)

    print(f"Attention maps saved → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ep    = args.epochs     or tc.get("epochs", 100)
    bs    = args.batch_size or tc.get("batch_size", 512)
    lr_   = tc.get("lr", 1e-3)
    nw    = tc.get("num_workers", 4)
    pw    = tc.get("persistent_workers", True)
    wd    = tc.get("weight_decay", 1e-4)
    gc    = tc.get("grad_clip", 1.0)

    lambda_kd   = args.lambda_kd
    temperature = args.temperature
    beta_crf    = args.beta_crf
    gamma_feat  = args.gamma_feature
    crf_T       = 0.07
    use_crf     = args.use_crf
    use_feat    = args.use_feature_kd
    tw_mode     = args.teacher_weight_mode

    # attention settings
    att_type        = args.attention_type
    att_layer_names = [s.strip() for s in args.attention_layers.split(",")]
    lambda_att      = args.lambda_att
    use_att_kd      = (att_type != "none") and (lambda_att > 0.0)
    att_loss_mode   = args.attn_loss_mode
    attn_w_parts    = [float(x) for x in args.attn_teacher_weights.split(",")]
    attn_w500       = attn_w_parts[0] / (sum(attn_w_parts) + 1e-8)
    attn_w100       = attn_w_parts[1] / (sum(attn_w_parts) + 1e-8)
    use_wide_kernel  = args.use_wide_kernel

    # teacher KD weights
    w_parts = [float(x) for x in args.weights.split(",")]
    w500, w100 = w_parts[0], w_parts[1]
    total_w = w500 + w100
    w500 /= total_w; w100 /= total_w

    # adaptive KD settings
    akd_mode           = args.akd_mode
    akd_gamma          = args.akd_gamma
    akd_min_weight     = args.akd_min_weight
    akd_detach_weight  = args.akd_detach_weight
    akd_norm_weight    = args.akd_normalize_weight
    label_corr_weight  = args.label_corr_weight
    dyn_w500_start     = args.dynamic_w500_start
    dyn_w500_end       = args.dynamic_w500_end

    # class reliability tensor [C]
    class_reliability: torch.Tensor | None = None
    if akd_mode == "class_reliability":
        src = args.class_reliability_source
        if src == "manual" and args.class_reliability_values:
            vals = [float(v) for v in args.class_reliability_values.split(",")]
            class_reliability = torch.tensor(vals, dtype=torch.float32)
        else:
            class_reliability = torch.ones(5, dtype=torch.float32)

    use_akd = (akd_mode != "base")

    # RKDLR settings
    method_mode       = getattr(args, "method_mode", "base")
    rkdlr_mode        = getattr(args, "rkdlr_mode", "gate_full")
    rkdlr_gamma       = getattr(args, "rkdlr_gamma", 1.0)
    rkdlr_min_weight  = getattr(args, "rkdlr_min_weight", 0.0)
    use_rkdlr         = (method_mode == "rkdlr")

    # CLRD settings
    use_clrd           = (method_mode == "clrd")
    lambda_clrd        = getattr(args, "lambda_clrd", 0.0)
    clrd_loss_type     = getattr(args, "clrd_loss", "kl")
    clrd_positive_only = getattr(args, "clrd_positive_only", True)
    clrd_cache_dir     = Path(getattr(args, "clrd_cache_dir",
                                      "dafd_mvkt/outputs/clrd_cache"))

    variant_id = args.variant_id or "F02opt"
    out_name   = args.output_name or f"student_II_50hz_f02_{variant_id}_seed{seed}"
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    ckpt_path = out_dir / f"{out_name}_best.pt"
    log_path  = out_dir / "logs" / f"{out_name}.csv"

    print(f"\n{'='*60}")
    print(f"F02 Student Tune: {variant_id}")
    print(f"w500={w500:.3f}  w100={w100:.3f}  T={temperature}  lambda_kd={lambda_kd}")
    print(f"use_crf={use_crf}  beta_crf={beta_crf}  use_feat={use_feat}  gamma={gamma_feat}")
    print(f"tw_mode={tw_mode}")
    print(f"attention_type={att_type}  layers={att_layer_names}  "
          f"lambda_att={lambda_att}  mode={att_loss_mode}")
    print(f"akd_mode={akd_mode}  gamma={akd_gamma}  min_w={akd_min_weight}  "
          f"detach={akd_detach_weight}  label_corr={label_corr_weight}")
    if use_rkdlr:
        print(f"RKDLR: mode={rkdlr_mode}  gamma={rkdlr_gamma}  min_w={rkdlr_min_weight}")
    if use_clrd:
        print(f"CLRD: lambda={lambda_clrd}  loss={clrd_loss_type}  "
              f"positive_only={clrd_positive_only}  cache_dir={clrd_cache_dir}")
    if akd_mode == "dynamic_weight":
        print(f"  dynamic_w500: {dyn_w500_start:.2f} -> {dyn_w500_end:.2f}")
    if akd_mode == "class_reliability" and class_reliability is not None:
        print(f"  class_reliability: {class_reliability.tolist()}")
    print(f"device={device}  seed={seed}  epochs={ep}  bs={bs}")
    print(f"output: {ckpt_path}")
    print(f"{'='*60}")

    # ── need_features flag for student / teachers ─────────────────────────────
    need_features = use_crf or use_feat or use_att_kd

    # ── frozen teachers ───────────────────────────────────────────────────────
    t500 = _build_model(in_channels=1).to(device)
    load_checkpoint(args.teacher_1l500_ckpt, t500, device)
    t500.eval()
    for p in t500.parameters(): p.requires_grad_(False)
    print(f"Teacher 1L500: {args.teacher_1l500_ckpt}")

    t100 = _build_model(in_channels=1).to(device)
    load_checkpoint(args.teacher_1l100_ckpt, t100, device)
    t100.eval()
    for p in t100.parameters(): p.requires_grad_(False)
    print(f"Teacher 1L100: {args.teacher_1l100_ckpt}")

    # ── 12-lead teacher (CLRD only) ───────────────────────────────────────────
    t12: nn.Module | None = None
    if use_clrd:
        t12 = _build_model(in_channels=12).to(device)
        load_checkpoint(args.teacher_12l100_ckpt, t12, device)
        t12.eval()
        for p in t12.parameters(): p.requires_grad_(False)
        print(f"Teacher 12L100: {args.teacher_12l100_ckpt}")

    # ── student ───────────────────────────────────────────────────────────────
    student = _build_model(
        in_channels=1,
        attention_type=att_type,
        attention_layers=att_layer_names if att_type != "none" else None,
        use_wide_kernel=use_wide_kernel,
    ).to(device)

    if args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        # strict=False: attention module weights are new, base encoder loaded
        load_encoder_init(student, args.student_init_ckpt, strict=False,
                          device=device)
        print(f"Student init: {args.student_init_ckpt}")
    else:
        print("Student: random init")

    n_p = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_att = sum(p.numel() for name, p in student.named_parameters()
                if "attentions" in name and p.requires_grad)
    print(f"Student params: {n_p/1e6:.3f}M  (attention: {n_att/1e3:.1f}K)")

    # ── feature KD module ─────────────────────────────────────────────────────
    feat_kd_fn: FeatureKDLoss | None = None
    if use_feat:
        feat_kd_fn = FeatureKDLoss(
            student_channels=student.feat_dim,
            ta_channels=t100.feat_dim,
            pool_size=128, normalize=True,
        ).to(device)
        print(f"FeatureKD: pool_size=128")

    # ── data ──────────────────────────────────────────────────────────────────
    lead = args.lead or cfg["data"].get("lead", "II")
    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pf = 4 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0),
                              prefetch_factor=pf)
    val_loader   = DataLoader(val_ds,  batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0),
                              prefetch_factor=pf)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(pw and nw > 0),
                              prefetch_factor=pf)

    if args.debug:
        first = next(iter(train_loader))
        print(f"\n[DEBUG] student: {first[STUDENT_KEY].shape}")
        print(f"[DEBUG] t1l500:  {first[T1L500_KEY].shape}")
        print(f"[DEBUG] t1l100:  {first[T1L100_KEY].shape}")
        xs = first[STUDENT_KEY][:2].to(device)
        with torch.no_grad():
            out = student(xs, return_features=True, return_attention=True)
        print(f"[DEBUG] logits:      {out['logits'].shape}")
        print(f"[DEBUG] feature_map: {out['feature_map'].shape}")
        print(f"[DEBUG] attentions:  {[(k, v.shape) for k,v in out.get('attentions',{}).items()]}")

    # ── CLRD: LeadImportanceHead + cache ─────────────────────────────────────
    lead_head: LeadImportanceHead | None = None
    clrd_cache: dict[int, torch.Tensor] = {}
    if use_clrd:
        feat_dim = student.feat_dim if hasattr(student, "feat_dim") else 512
        lead_head = LeadImportanceHead(feat_dim=feat_dim, num_classes=5,
                                       num_leads=12).to(device)
        clrd_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = clrd_cache_dir / f"clrd_cache_lead{lead}_seed{seed}.pt"
        if cache_file.exists():
            print(f"Loading CLRD cache: {cache_file}")
            clrd_cache = torch.load(cache_file, map_location="cpu")
            print(f"  {len(clrd_cache)} records loaded")
        else:
            print("Building CLRD lead importance cache …")
            # use train+val+test loaders to build full cache
            for _split_loader in (train_loader, val_loader, test_loader):
                split_cache = _build_lead_importance_cache(
                    t12, _split_loader, device, temperature=temperature)
                clrd_cache.update(split_cache)
            torch.save(clrd_cache, cache_file)
            print(f"  Saved {len(clrd_cache)} records → {cache_file}")

    # ── optimizer ─────────────────────────────────────────────────────────────
    params = list(student.parameters())
    if feat_kd_fn is not None:
        params += list(feat_kd_fn.parameters())
    if lead_head is not None:
        params += list(lead_head.parameters())

    bce_fn    = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params, lr=lr_, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc  = 0.0
    log_rows  = []
    eps_val   = 1e-6

    for epoch in range(1, ep + 1):
        student.train()
        if feat_kd_fn is not None:
            feat_kd_fn.train()

        ep_bce = ep_kd = ep_crf = ep_feat = ep_att = ep_corr = ep_tot = 0.0
        ep_akd_w = 0.0
        ep_r = ep_r500 = ep_r100 = 0.0
        ep_clrd = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{ep}",
                          dynamic_ncols=True, leave=False):
            y    = batch["y"].to(device)
            s_in = batch[STUDENT_KEY].to(device)

            s_out    = student(s_in,
                               return_features=(need_features or use_clrd),
                               return_attention=use_att_kd)
            s_logits = s_out["logits"]

            with torch.no_grad():
                x500 = batch[T1L500_KEY].to(device)   # [B,1,5000]
                x100 = batch[T1L100_KEY].to(device)   # [B,1,1000]
                t500_out = t500(x500, return_features=use_att_kd)
                t100_out = t100(x100, return_features=need_features)
                l500 = t500_out["logits"]
                l100 = t100_out["logits"]
                if use_rkdlr:
                    x500_deg = _degrade_signal(x500, 500)   # [B,1,5000]
                    x100_deg = _degrade_signal(x100, 500)   # [B,1,1000]
                    l500_deg = t500(x500_deg)["logits"]
                    l100_deg = t100(x100_deg)["logits"]

            # ── KD loss (rkdlr / adaptive / base) ─────────────────────────────
            if use_rkdlr:
                loss_kd, rkdlr_info = _compute_rkdlr_loss(
                    s_logits=s_logits,
                    l500_full=l500, l100_full=l100,
                    l500_deg=l500_deg, l100_deg=l100_deg,
                    w500=w500, w100=w100,
                    temperature=temperature,
                    rkdlr_mode=rkdlr_mode,
                    rkdlr_gamma=rkdlr_gamma,
                    rkdlr_min_weight=rkdlr_min_weight,
                )
                loss_corr_val = s_logits.new_zeros(1)
                ep_r    += rkdlr_info["mean_recoverability"]
                ep_r500 += rkdlr_info["mean_r500"]
                ep_r100 += rkdlr_info["mean_r100"]
            elif use_akd:
                loss_kd, loss_corr_val, akd_info = _compute_adaptive_kd_loss(
                    s_logits=s_logits,
                    l500=l500, l100=l100,
                    w500=w500, w100=w100,
                    temperature=temperature,
                    akd_mode=akd_mode,
                    akd_gamma=akd_gamma,
                    akd_min_weight=akd_min_weight,
                    akd_detach_weight=akd_detach_weight,
                    akd_normalize_weight=akd_norm_weight,
                    class_reliability=class_reliability,
                    label_corr_weight=label_corr_weight,
                    current_epoch=epoch,
                    total_epochs=ep,
                    dynamic_w500_start=dyn_w500_start,
                    dynamic_w500_end=dyn_w500_end,
                )
                ep_akd_w += akd_info.get("mean_akd_weight", 0.0)
            else:
                # base: legacy tw_mode path (backward compat)
                if tw_mode == "confidence":
                    with torch.no_grad():
                        p500 = torch.sigmoid(l500)
                        p100 = torch.sigmoid(l100)
                        c500 = (p500 - 0.5).abs()
                        c100 = (p100 - 0.5).abs()
                        denom = c500 + c100 + eps_val
                        w5_s = (c500 / denom).mean().item()
                        w1_s = (c100 / denom).mean().item()
                    kd500 = multi_label_kd_loss(s_logits, l500, temperature)
                    kd100 = multi_label_kd_loss(s_logits, l100, temperature)
                    loss_kd = w5_s * kd500 + w1_s * kd100
                elif tw_mode == "agreement":
                    with torch.no_grad():
                        scale = (1.0 - (torch.sigmoid(l500)
                                        - torch.sigmoid(l100)).abs()).mean().item()
                    kd500 = multi_label_kd_loss(s_logits, l500, temperature)
                    kd100 = multi_label_kd_loss(s_logits, l100, temperature)
                    loss_kd = scale * (w500 * kd500 + w100 * kd100)
                else:
                    kd500 = multi_label_kd_loss(s_logits, l500, temperature)
                    kd100 = multi_label_kd_loss(s_logits, l100, temperature)
                    loss_kd = w500 * kd500 + w100 * kd100
                loss_corr_val = s_logits.new_zeros(1)

            loss_bce = bce_fn(s_logits, y)
            loss = loss_bce + lambda_kd * loss_kd
            if use_akd and akd_mode == "label_correlation" and label_corr_weight > 0.0:
                loss = loss + label_corr_weight * loss_corr_val

            # ── CRF ───────────────────────────────────────────────────────────
            loss_crf_val = torch.zeros(1, device=device)
            if use_crf:
                loss_crf_val = cross_rate_contrastive_loss(
                    s_out["pooled"],   t100_out["pooled"],
                    s_out["proj"],     t100_out["proj"],
                    temperature=crf_T,
                )
                loss = loss + beta_crf * loss_crf_val

            # ── FeatureKD ─────────────────────────────────────────────────────
            loss_feat_val = torch.zeros(1, device=device)
            if use_feat and feat_kd_fn is not None:
                loss_feat_val = feat_kd_fn(
                    s_out["feature_map"], t100_out["feature_map"])
                loss = loss + gamma_feat * loss_feat_val

            # ── CLRD ──────────────────────────────────────────────────────────
            loss_clrd_val = torch.zeros(1, device=device)
            if use_clrd and lead_head is not None and lambda_clrd > 0.0:
                rids = batch["record_id"].tolist()
                A_teacher_list = [clrd_cache.get(int(r)) for r in rids]
                if all(a is not None for a in A_teacher_list):
                    A_teacher = torch.stack(A_teacher_list).to(device)  # [B,5,12]
                    pooled = s_out.get("pooled")
                    if pooled is None:
                        pooled = s_out["feature_map"].mean(dim=-1)      # [B, feat]
                    A_student = lead_head(pooled)                        # [B,5,12]
                    loss_clrd_val = _compute_clrd_loss(
                        A_student, A_teacher.detach(), y,
                        clrd_loss=clrd_loss_type,
                        positive_only=clrd_positive_only,
                    )
                    loss = loss + lambda_clrd * loss_clrd_val
            ep_clrd += (loss_clrd_val.item()
                        if isinstance(loss_clrd_val, torch.Tensor)
                        else 0.0)

            # ── Attention KD ──────────────────────────────────────────────────
            loss_att_val = torch.zeros(1, device=device)
            if use_att_kd:
                att_losses = []
                s_atts = s_out.get("attentions", {})
                for lay in att_layer_names:
                    if lay not in s_atts:
                        continue
                    s_att = s_atts[lay]                     # [B, 1, L_s]
                    t_sal = make_dual_teacher_saliency(
                        t500_out["feature_map"],
                        t100_out["feature_map"],
                        target_len=s_att.shape[-1],
                        w500=attn_w500,
                        w100=attn_w100,
                    )
                    att_losses.append(
                        attention_kd_loss(s_att, t_sal, mode=att_loss_mode)
                    )
                if att_losses:
                    loss_att_val = sum(att_losses) / len(att_losses)
                    loss = loss + lambda_att * loss_att_val

            optimizer.zero_grad()
            loss.backward()
            if gc > 0:
                nn.utils.clip_grad_norm_(params, gc)
            optimizer.step()

            ep_bce  += loss_bce.item()
            ep_kd   += loss_kd.item()
            ep_crf  += loss_crf_val.item()
            ep_feat += loss_feat_val.item()
            ep_att  += (loss_att_val.item()
                        if isinstance(loss_att_val, torch.Tensor)
                        else loss_att_val)
            ep_corr += (loss_corr_val.item()
                        if isinstance(loss_corr_val, torch.Tensor)
                        else 0.0)
            ep_tot  += loss.item()
            n_batches += 1

        scheduler.step()

        val_probs, val_labels, _ = _run_eval(student, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = val_metrics["macro_auc"]
        val_f1  = val_metrics.get("macro_f1", 0.0)
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, student, epoch, {"val_auc": val_auc},
                            {"variant": variant_id, "w500": w500, "w100": w100,
                             "temperature": temperature,
                             "attention_type": att_type,
                             "akd_mode": akd_mode})

        n = n_batches
        att_str   = f" att={ep_att/n:.4f}"  if use_att_kd else ""
        corr_str  = f" corr={ep_corr/n:.5f}" if (use_akd and akd_mode == "label_correlation") else ""
        akdw_str  = f" akd_w={ep_akd_w/n:.3f}" if use_akd else ""
        rkdlr_str = f" r={ep_r/n:.3f} r500={ep_r500/n:.3f} r100={ep_r100/n:.3f}" if use_rkdlr else ""
        clrd_str  = f" clrd={ep_clrd/n:.5f}" if use_clrd else ""

        # dynamic weight current values
        if use_akd and akd_mode == "dynamic_weight":
            frac = (epoch - 1) / max(ep - 1, 1)
            cur_w5 = dyn_w500_start + frac * (dyn_w500_end - dyn_w500_start)
            akdw_str += f" w500={cur_w5:.3f}"

        print(f"Ep {epoch:3d} | bce={ep_bce/n:.4f} kd={ep_kd/n:.4f} "
              f"crf={ep_crf/n:.4f} feat={ep_feat/n:.4f}{att_str}{corr_str}{akdw_str}{rkdlr_str}{clrd_str} "
              f"tot={ep_tot/n:.4f} | val_auc={val_auc:.4f} val_f1={val_f1:.4f} "
              f"{'★' if is_best else ''}")

        log_rows.append({
            "epoch":              epoch,
            "bce":                round(ep_bce/n, 5),
            "kd":                 round(ep_kd/n, 5),
            "crf":                round(ep_crf/n, 5),
            "feat":               round(ep_feat/n, 5),
            "att":                round(ep_att/n, 5),
            "label_corr":         round(ep_corr/n, 6),
            "akd_weight":         round(ep_akd_w/n, 4),
            "mean_recoverability":round(ep_r/n, 5) if use_rkdlr else "",
            "mean_r500":          round(ep_r500/n, 5) if use_rkdlr else "",
            "mean_r100":          round(ep_r100/n, 5) if use_rkdlr else "",
            "clrd":               round(ep_clrd/n, 6) if use_clrd else "",
            "total":              round(ep_tot/n, 5),
            "val_auc":            round(val_auc, 5),
            "val_f1":             round(val_f1, 5),
            "best_val_auc":       round(best_auc, 5),
        })

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    # ── final evaluation ──────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f}) …")
    load_checkpoint(ckpt_path, student, device)

    val_probs, val_labels, _          = _run_eval(student, val_loader,  device)
    grid       = np.arange(0.05, 0.96, 0.01)
    thresholds = find_best_thresholds(val_labels, val_probs, grid=grid)

    test_probs, test_labels, test_ids = _run_eval(student, test_loader, device)
    m05    = compute_metrics(test_labels, test_probs, threshold=0.5)
    m_tune = compute_metrics(test_labels, test_probs, threshold=thresholds)

    # dynamic weight final values
    if akd_mode == "dynamic_weight":
        final_w500 = dyn_w500_end
        final_w100 = 1.0 - dyn_w500_end
    else:
        final_w500, final_w100 = w500, w100

    metrics = {
        "output_name":      out_name,
        "variant_id":       variant_id,
        "w500":             w500, "w100": w100,
        "temperature":      temperature,
        "use_crf":          use_crf, "beta_crf": beta_crf,
        "use_feature_kd":   use_feat, "gamma_feature": gamma_feat,
        "tw_mode":          tw_mode,
        "attention_type":   att_type,
        "attention_layers": ",".join(att_layer_names),
        "lambda_att":       lambda_att,
        "attn_loss_mode":   att_loss_mode,
        # adaptive KD fields
        "akd_mode":              akd_mode,
        "akd_gamma":             akd_gamma,
        "akd_min_weight":        akd_min_weight,
        "label_corr_weight":     label_corr_weight,
        # RKDLR fields
        "method_mode":           method_mode,
        "rkdlr_mode":            rkdlr_mode if use_rkdlr else None,
        "rkdlr_gamma":           rkdlr_gamma if use_rkdlr else None,
        "rkdlr_min_weight":      rkdlr_min_weight if use_rkdlr else None,
        "mean_recoverability":   round(ep_r / max(n_batches, 1), 5) if use_rkdlr else None,
        "mean_r500":             round(ep_r500 / max(n_batches, 1), 5) if use_rkdlr else None,
        "mean_r100":             round(ep_r100 / max(n_batches, 1), 5) if use_rkdlr else None,
        # CLRD fields
        "lambda_clrd":           lambda_clrd if use_clrd else None,
        "clrd_loss_type":        clrd_loss_type if use_clrd else None,
        "clrd_positive_only":    clrd_positive_only if use_clrd else None,
        "class_reliability_values": (
            args.class_reliability_values
            if akd_mode == "class_reliability" else None
        ),
        "dynamic_w500_start":    dyn_w500_start if akd_mode == "dynamic_weight" else None,
        "dynamic_w500_end":      dyn_w500_end   if akd_mode == "dynamic_weight" else None,
        "final_w500":            round(final_w500, 4),
        "final_w100":            round(final_w100, 4),
        # performance
        "macro_auc":        round(m05["macro_auc"], 5),
        "macro_f1_0_5":     round(m05.get("macro_f1", 0), 5),
        "macro_f1_tuned":   round(m_tune.get("macro_f1", 0), 5),
        "val_auc_best":     round(best_auc, 5),
        "per_class_thresholds": {
            SUPERCLASSES[i]: round(float(thresholds[i]), 4) for i in range(5)
        },
        "_meta": {"seed": seed, "lambda_kd": lambda_kd,
                  "n_params": n_p},
    }
    for c in SUPERCLASSES:
        metrics[f"auc_{c}"]      = round(m05.get(f"auc_{c}", float("nan")), 5)
        metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    val_m05    = compute_metrics(val_labels, val_probs, threshold=0.5)
    val_m_tune = compute_metrics(val_labels, val_probs, threshold=thresholds)
    with open(out_dir / f"metrics_val_{out_name}.json", "w") as f:
        json.dump({"output_name": out_name,
                   "macro_auc":      round(val_m05["macro_auc"], 5),
                   "macro_f1_tuned": round(val_m_tune.get("macro_f1", 0), 5)},
                  f, indent=2)
    with open(out_dir / f"metrics_test_{out_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd
    preds = (test_probs >= thresholds[None]).astype(int)
    rows  = [{"record_id": rid,
               **{f"y_{c}":    int(test_labels[i, ci])
                  for ci, c in enumerate(SUPERCLASSES)},
               **{f"prob_{c}": round(float(test_probs[i, ci]), 5)
                  for ci, c in enumerate(SUPERCLASSES)},
               **{f"pred_{c}": int(preds[i, ci])
                  for ci, c in enumerate(SUPERCLASSES)}}
             for i, rid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(
        out_dir / f"predictions_test_{out_name}.csv", index=False)

    # ── classwise CSV ─────────────────────────────────────────────────────────
    cw_rows = []
    for ci, c in enumerate(SUPERCLASSES):
        cw_rows.append({
            "class":         c,
            "auc":           round(m05.get(f"auc_{c}", float("nan")), 5),
            "f1_at_0.5":     round(m05.get(f"f1_{c}", float("nan")), 5),
            "f1_tuned":      round(m_tune.get(f"f1_{c}", float("nan")), 5),
            "threshold":     round(float(thresholds[ci]), 4),
        })
    import pandas as _pd2
    _pd2.DataFrame(cw_rows).to_csv(
        out_dir / f"classwise_{out_name}.csv", index=False)

    # ── summary CSV (adaptive KD columns) ─────────────────────────────────────
    CLASSES = SUPERCLASSES
    summary_row = {
        "variant_id":              variant_id,
        "method_mode":             method_mode,
        "rkdlr_mode":              rkdlr_mode if use_rkdlr else "",
        "rkdlr_gamma":             rkdlr_gamma if use_rkdlr else "",
        "rkdlr_min_weight":        rkdlr_min_weight if use_rkdlr else "",
        "mean_recoverability":     round(ep_r / max(n_batches, 1), 5) if use_rkdlr else "",
        "mean_r500":               round(ep_r500 / max(n_batches, 1), 5) if use_rkdlr else "",
        "mean_r100":               round(ep_r100 / max(n_batches, 1), 5) if use_rkdlr else "",
        "lambda_clrd":             lambda_clrd if use_clrd else "",
        "clrd_loss_type":          clrd_loss_type if use_clrd else "",
        "akd_mode":                akd_mode,
        "akd_gamma":               akd_gamma,
        "akd_min_weight":          akd_min_weight,
        "label_corr_weight":       label_corr_weight,
        "class_reliability_values": (
            args.class_reliability_values
            if akd_mode == "class_reliability" else ""),
        "dynamic_w500_start":      dyn_w500_start if akd_mode == "dynamic_weight" else "",
        "dynamic_w500_end":        dyn_w500_end   if akd_mode == "dynamic_weight" else "",
        "AUC_macro":               metrics["macro_auc"],
        "F1_tuned":                metrics["macro_f1_tuned"],
        "F1_at_0.5":               metrics["macro_f1_0_5"],
        **{f"{c}_AUC":  metrics[f"auc_{c}"]      for c in CLASSES},
        **{f"{c}_F1":   metrics[f"f1_tuned_{c}"] for c in CLASSES},
        "best_epoch":              best_auc,
        "seed":                    seed,
        "params":                  round(n_p / 1e6, 3),
        "notes":                   "",
        # backward compat fields
        "temperature":             temperature,
        "w500":                    round(w500, 3),
        "w100":                    round(w100, 3),
    }
    _update_summary(out_dir, summary_row)

    auc = metrics["macro_auc"]; f1t = metrics["macro_f1_tuned"]
    print(f"\nAUC={auc:.4f}  F1_tuned={f1t:.4f}")
    print(f"vs F02:  {auc-F02_AUC:+.4f} AUC  {f1t-F02_F1:+.4f} F1")
    print(f"vs MVKT: {auc-MVKT_AUC:+.4f} AUC")
    print(f"Beats F02:  {'YES ✓' if auc>F02_AUC else 'no'}  "
          f"MVKT AUC: {'YES ✓' if auc>MVKT_AUC else 'no'}")
    print(f"Saved: {out_dir}/metrics_test_{out_name}.json")

    # ── attention map visualization ───────────────────────────────────────────
    if args.save_attention_maps and att_type != "none":
        print("\nSaving attention maps …")
        _save_attention_maps(
            student, t500, t100, test_loader, device,
            variant_id=variant_id,
            out_dir=out_dir,
            vis_n=args.attention_vis_n,
            att_layer_names=att_layer_names,
            attn_w500=attn_w500,
            attn_w100=attn_w100,
        )


# ─────────────────────────────────────────────────────────────────────────────
# argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # ── core ──────────────────────────────────────────────────────────────────
    p.add_argument("--config",             default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",           required=True)
    p.add_argument("--lead",               default="II")
    p.add_argument("--teacher_1l500_ckpt", required=True)
    p.add_argument("--teacher_1l100_ckpt", required=True)
    p.add_argument("--student_init_ckpt",  default=None)
    p.add_argument("--weights",            default="0.5,0.5",
                   help="comma-separated w500,w100 (auto-normalised)")
    p.add_argument("--lambda_kd",          type=float, default=1.0)
    p.add_argument("--temperature",        type=float, default=2.0)
    # ── CRF ───────────────────────────────────────────────────────────────────
    p.add_argument("--use_crf",            action="store_true")
    p.add_argument("--beta_crf",           type=float, default=0.05)
    # ── FeatureKD ─────────────────────────────────────────────────────────────
    p.add_argument("--use_feature_kd",     action="store_true")
    p.add_argument("--gamma_feature",      type=float, default=0.10)
    # ── confidence weighting ──────────────────────────────────────────────────
    p.add_argument("--teacher_weight_mode", default="fixed",
                   choices=["fixed", "confidence", "agreement"])
    # ── attention (TG-MTA) ────────────────────────────────────────────────────
    p.add_argument("--attention_type",      default="none",
                   choices=["none", "se", "cbam", "tgmta"])
    p.add_argument("--attention_layers",    default="layer3,layer4",
                   help="comma-separated stage names, e.g. layer3,layer4")
    p.add_argument("--lambda_att",          type=float, default=0.0,
                   help="weight for attention KD loss (0 = disabled)")
    p.add_argument("--attn_teacher_weights", default="0.60,0.40",
                   help="w500,w100 for dual teacher saliency (auto-normalised)")
    p.add_argument("--attn_loss_mode",      default="mse_prob",
                   choices=["mse_prob", "kl"])
    p.add_argument("--use_wide_kernel",      action="store_true",
                   help="use wide kernel (k=31) branch in TGMTAModule")
    p.add_argument("--save_attention_maps", action="store_true")
    p.add_argument("--attention_vis_n",     type=int, default=5)
    # ── adaptive KD ───────────────────────────────────────────────────────────
    p.add_argument("--akd_mode",            default="base",
                   choices=_AKD_MODES,
                   help="adaptive KD mode (default=base: identical to F02T01)")
    p.add_argument("--akd_gamma",           type=float, default=1.0,
                   help="sharpness exponent for confidence/agreement weights")
    p.add_argument("--akd_min_weight",      type=float, default=0.0,
                   help="minimum adaptive weight floor in [0,1)")
    p.add_argument("--akd_detach_weight",   action="store_true",
                   help="detach adaptive weight from computation graph")
    p.add_argument("--akd_normalize_weight", action="store_true",
                   help="normalize adaptive weight to mean=1")
    p.add_argument("--label_corr_weight",   type=float, default=0.0,
                   help="weight for label-correlation loss")
    p.add_argument("--dynamic_w500_start",  type=float, default=0.50,
                   help="initial w500 for dynamic_weight schedule")
    p.add_argument("--dynamic_w500_end",    type=float, default=0.60,
                   help="final w500 for dynamic_weight schedule")
    p.add_argument("--class_reliability_values", default=None,
                   help="comma-separated per-class reliability weights, e.g. 1.0,1.0,1.0,1.0,0.75")
    p.add_argument("--class_reliability_source", default="uniform",
                   choices=["manual", "teacher_auc", "uniform"])
    # ── RKDLR / CLRD ──────────────────────────────────────────────────────────
    p.add_argument("--method_mode",       default="base",
                   choices=["base", "rkdlr", "clrd"],
                   help="'base'=F02T01 identical; 'rkdlr'=recoverability KD; 'clrd'=lead importance KD")
    p.add_argument("--rkdlr_mode",        default="gate_full",
                   choices=_RKDLR_MODES)
    p.add_argument("--rkdlr_gamma",       type=float, default=1.0)
    p.add_argument("--rkdlr_min_weight",  type=float, default=0.0)
    # ── CLRD ──────────────────────────────────────────────────────────────────
    p.add_argument("--teacher_12l100_ckpt", default=None,
                   help="12-lead 100Hz teacher checkpoint (required for --method_mode clrd)")
    p.add_argument("--lambda_clrd",       type=float, default=0.0,
                   help="weight for CLRD lead importance KD loss")
    p.add_argument("--clrd_loss",         default="kl", choices=["kl", "mse"],
                   help="CLRD loss type: kl or mse")
    p.add_argument("--clrd_positive_only", action="store_true",
                   help="only penalize positive (label=1) class/sample pairs in CLRD")
    p.add_argument("--clrd_cache_dir",    default="dafd_mvkt/outputs/clrd_cache",
                   help="directory for pre-computed lead importance cache")
    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--epochs",             type=int,   default=None)
    p.add_argument("--batch_size",         type=int,   default=None)
    p.add_argument("--variant_id",         default=None)
    p.add_argument("--output_name",        default=None)
    p.add_argument("--output_dir",         default="dafd_mvkt/outputs/f02_opt")
    p.add_argument("--debug",              action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
