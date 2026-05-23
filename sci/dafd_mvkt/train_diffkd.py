"""
DiffKD: Diffusion-based Feature Knowledge Distillation.

Training variants (run via --variant_id):
  DKD00_dryrun : λ_diff=0.0, dry-run to verify F02T01 reproduced
  DKD01        : Naive DiffKD (λ_diff=0.5, v-pred, phase1→phase2→phase3)
  DKD03        : Unconditional diffusion ablation (no student cond)
  DKD05        : layer3 feature instead of layer4
  DKD07        : No phase1 pretrain (joint from scratch)
  DKD09        : epsilon-prediction instead of v-prediction

Fixed (matching F02T01):
  w500=0.60, w100=0.40, T=1.5, BCE + MKD loss
  seed=0, adaptive_50hz.yaml, metric/threshold code identical

Outputs (per variant):
  dafd_mvkt/outputs/diffkd/<variant_id>/
    metrics_test.json   best.pt   train_log.txt   config_snapshot.json

Entry point:
  python dafd_mvkt/train_diffkd.py --variant_id DKD00_dryrun [options]
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))        # so "models.*", "data.*", etc. resolve

from data.ptbxl_multiteacher_dataset import PTBXLMultiTeacherDataset
from diffkd.unet1d       import build_unet
from diffkd.scheduler    import DDPMScheduler
from diffkd.feature_hook import DualTeacherFeatureExtractor
from losses.mkd          import multi_label_kd_loss
from models.resnet1d     import ResNet1d
from utils.checkpoint    import load_checkpoint, save_checkpoint
from utils.metrics       import compute_metrics, find_best_thresholds
from utils.seed          import set_seed

LOCAL_F02T01_AUC = 0.84644
LOCAL_F02T01_F1  = 0.62544
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_resnet(in_ch: int = 1) -> ResNet1d:
    return ResNet1d(in_channels=in_ch, num_classes=5,
                    layers=[3, 4, 6, 3], base_channels=64,
                    proj_dim=128, dropout=0.0)


@torch.no_grad()
def _run_eval(
    student: nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    student.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        x   = batch["student_x"].to(device)
        out = student(x, return_features=False)
        all_logits.append(out["logits"].cpu().float())
        all_labels.append(batch["y"].cpu().float())
        all_ids.extend(batch["record_id"].tolist())
    logits = torch.cat(all_logits).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, torch.cat(all_labels).numpy(), all_ids


def _save_metrics(out_dir: Path, name: str, metrics: dict, extra: dict) -> None:
    payload = {**metrics, **extra}
    with open(out_dir / name, "w") as f:
        json.dump(payload, f, indent=2)


# ── diffusion loss ────────────────────────────────────────────────────────────

def _diffusion_loss(
    diffusion:   nn.Module,
    scheduler:   DDPMScheduler,
    feat_s:      torch.Tensor,   # [B, C, L]  student feat (cond)
    feat_t:      torch.Tensor,   # [B, C, L]  teacher feat (target)
    unconditional: bool = False,
) -> torch.Tensor:
    """Single-step diffusion training loss."""
    B = feat_t.shape[0]
    t = torch.randint(0, scheduler.num_timesteps, (B,), device=feat_t.device)
    noise = torch.randn_like(feat_t)
    x_t, _ = scheduler.q_sample(feat_t, t, noise)

    cond = torch.zeros_like(feat_s) if unconditional else feat_s
    pred = diffusion(x_t, t, cond)
    return scheduler.training_loss(pred, feat_t, noise, t)


# ── training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    t_start = time.time()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tc = cfg["training"]

    seed = args.seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── config resolution ────────────────────────────────────────────────────
    ep_total  = args.epochs  or tc.get("epochs", 100)
    bs        = args.batch_size or tc.get("batch_size", 512)
    lr_       = tc.get("lr", 1e-3)
    nw        = tc.get("num_workers", 4)
    pw        = tc.get("persistent_workers", True)
    wd        = tc.get("weight_decay", 1e-4)
    gc        = tc.get("grad_clip", 1.0)

    # phase split (only DKD01/07 use multi-phase; others do 0+ep_total+0)
    ep_phase1 = args.phase1_epochs   # diffusion-only pretrain
    ep_phase2 = args.phase2_epochs   # joint
    ep_phase3 = args.phase3_epochs   # student-only (freeze diffusion)
    if ep_phase1 + ep_phase2 + ep_phase3 == 0:
        # default: all epochs in phase2
        ep_phase2 = ep_total

    lambda_kd   = args.lambda_kd
    lambda_diff = args.lambda_diff
    temperature = args.temperature
    w_parts = [float(x) for x in args.weights.split(",")]
    w500 = w_parts[0] / sum(w_parts)
    w100 = w_parts[1] / sum(w_parts)

    feat_layer   = args.feat_layer       # "layer4" | "layer3"
    unconditional = args.unconditional   # DKD03: no cond
    pred_type    = args.pred_type        # "v_prediction" | "epsilon"
    cond_type    = args.cond_type        # "concat" | "attn"
    diff_steps   = args.diff_steps

    variant_id = args.variant_id
    out_dir    = Path(args.output_dir) / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path  = out_dir / "train_log.txt"
    ckpt_path = out_dir / "best.pt"

    # feature dim depends on layer
    if feat_layer == "layer4":
        feat_dim, feat_L = 512, 16
    else:  # layer3
        feat_dim, feat_L = 256, 32

    print(f"\n{'='*65}")
    print(f"DiffKD variant: {variant_id}")
    print(f"feat={feat_layer}({feat_dim}×{feat_L})  cond={cond_type}  pred={pred_type}")
    print(f"λ_diff={lambda_diff}  λ_kd={lambda_kd}  T={temperature}")
    print(f"w500={w500:.3f}  w100={w100:.3f}  diff_steps={diff_steps}")
    print(f"phases: p1={ep_phase1} p2={ep_phase2} p3={ep_phase3}")
    print(f"unconditional={unconditional}  seed={seed}  device={device}")
    print(f"output: {out_dir}")
    print(f"{'='*65}")

    # ── save config ──────────────────────────────────────────────────────────
    snap = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    snap["__resolved"] = dict(w500=w500, w100=w100, ep_phase1=ep_phase1,
                              ep_phase2=ep_phase2, ep_phase3=ep_phase3,
                              feat_dim=feat_dim, feat_L=feat_L)
    with open(out_dir / "config_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)

    # ── teachers ─────────────────────────────────────────────────────────────
    def _load_teacher(ckpt: str, in_ch: int = 1) -> ResNet1d:
        m = _build_resnet(in_ch).to(device)
        load_checkpoint(ckpt, m, device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    t500 = _load_teacher(args.teacher_1l500_ckpt)
    t100 = _load_teacher(args.teacher_1l100_ckpt)
    print(f"T500: {args.teacher_1l500_ckpt}")
    print(f"T100: {args.teacher_1l100_ckpt}")

    # ── student ──────────────────────────────────────────────────────────────
    student = _build_resnet().to(device)
    ckpt_meta = load_checkpoint(args.f02t01_ckpt, student, device)
    print(f"Student init (F02T01): {args.f02t01_ckpt}")
    print(f"  checkpoint epoch={ckpt_meta.get('epoch','?')}  "
          f"val_auc={ckpt_meta.get('metrics',{}).get('val_auc','?')}")

    # ── diffusion model ───────────────────────────────────────────────────────
    unet = build_unet(
        feature_dim=feat_dim,
        base_channels=args.unet_base_ch,
        mults=tuple(int(x) for x in args.unet_mults.split(",")),
        cond_type=cond_type,
    ).to(device)
    scheduler = DDPMScheduler(
        num_timesteps=diff_steps,
        prediction_type=pred_type,
    ).to(device)
    n_diff = sum(p.numel() for p in unet.parameters())
    n_stu  = sum(p.numel() for p in student.parameters())
    print(f"UNet params: {n_diff/1e6:.3f}M   Student params: {n_stu/1e6:.3f}M")

    # ── feature hooks ─────────────────────────────────────────────────────────
    feat_extractor = DualTeacherFeatureExtractor(
        student, t500, t100, layer=feat_layer)

    # ── data ─────────────────────────────────────────────────────────────────
    lead = "II"
    train_ds = PTBXLMultiTeacherDataset(args.data_dir, split="train", lead=lead)
    val_ds   = PTBXLMultiTeacherDataset(args.data_dir, split="val",   lead=lead)
    test_ds  = PTBXLMultiTeacherDataset(args.data_dir, split="test",  lead=lead)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    pf = 4 if nw > 0 else None
    kw = dict(num_workers=nw, pin_memory=True,
              persistent_workers=(pw and nw > 0), prefetch_factor=pf)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, **kw)

    # ── sanity: run 1 batch, check shapes and losses ─────────────────────────
    print("\n[sanity check]")
    _b = next(iter(train_loader))
    _xs  = _b["student_x"].to(device)
    _x5  = _b["teacher_1l_500"].to(device)
    _x1  = _b["teacher_1l_100"].to(device)
    _y   = _b["y"].to(device)

    with torch.no_grad():
        _so  = student(_xs, return_features=True)
        _l5  = t500(_x5)["logits"]
        _l1  = t100(_x1)["logits"]
        _fs, _ft5, _ft1 = feat_extractor.get()

        _bce  = nn.BCEWithLogitsLoss()(_so["logits"], _y)
        _kd5  = multi_label_kd_loss(_so["logits"], _l5, temperature)
        _kd1  = multi_label_kd_loss(_so["logits"], _l1, temperature)

        _t_batch = torch.randint(0, diff_steps, (_xs.shape[0],), device=device)
        _noise   = torch.randn_like(_fs)
        _xt, _   = scheduler.q_sample(_fs, _t_batch, _noise)
        _cond    = torch.zeros_like(_fs) if unconditional else _fs
        _vpred   = unet(_xt, _t_batch, _cond)
        _dl      = scheduler.training_loss(_vpred, _fs, _noise, _t_batch)

    print(f"  feat_student: {tuple(_fs.shape)}  feat_t500_pooled: {tuple(_ft5.shape)}")
    print(f"  BCE={_bce:.4f}  KD500={_kd5:.4f}  KD100={_kd1:.4f}  DiffLoss={_dl:.4f}")
    print(f"  student logits: mean={_so['logits'].mean():.3f} std={_so['logits'].std():.3f}")
    assert not _vpred.isnan().any(), "NaN in UNet output!"
    print("[sanity] OK\n")

    # ── optimizers ───────────────────────────────────────────────────────────
    bce_fn = nn.BCEWithLogitsLoss()

    opt_student = torch.optim.AdamW(student.parameters(), lr=lr_, weight_decay=wd)
    opt_diff    = torch.optim.AdamW(unet.parameters(),    lr=lr_, weight_decay=wd)

    total_ep = ep_phase1 + ep_phase2 + ep_phase3
    sched_student = torch.optim.lr_scheduler.CosineAnnealingLR(opt_student, T_max=total_ep)
    sched_diff    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_diff,    T_max=total_ep)

    # ── training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    log_rows: list[dict] = []

    def _phase_of(epoch: int) -> int:
        if epoch <= ep_phase1:              return 1
        if epoch <= ep_phase1 + ep_phase2:  return 2
        return 3

    log_file = open(log_path, "w")

    for epoch in range(1, total_ep + 1):
        phase = _phase_of(epoch)
        student.train()
        unet.train()

        ep_bce = ep_kd = ep_diff = ep_tot = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{total_ep} [P{phase}]",
                          dynamic_ncols=True, leave=False):
            y   = batch["y"].to(device)
            x_s = batch["student_x"].to(device)
            x_500 = batch["teacher_1l_500"].to(device)
            x_100 = batch["teacher_1l_100"].to(device)

            # teacher forward (always frozen, no grad)
            with torch.no_grad():
                l500 = t500(x_500)["logits"]
                l100 = t100(x_100)["logits"]

            # student forward
            need_feat = (lambda_diff > 0.0 and phase in (2, 3))
            s_out  = student(x_s, return_features=need_feat)
            s_log  = s_out["logits"]

            loss_bce = bce_fn(s_log, y)
            loss_kd  = (w500 * multi_label_kd_loss(s_log, l500, temperature)
                      + w100 * multi_label_kd_loss(s_log, l100, temperature))

            # diffusion loss
            loss_diff = torch.zeros(1, device=device)
            if lambda_diff > 0.0:
                # get features via hooks (triggered by above student/teacher fwds)
                fs, ft5, ft1 = feat_extractor.get()
                # use average of teacher features as diffusion target
                feat_teacher = (w500 * ft5 + w100 * ft1).detach()

                if phase == 1:
                    # diffusion pretrain: only diff loss, student frozen
                    with torch.no_grad():
                        pass   # fs already computed
                    loss_diff = _diffusion_loss(unet, scheduler, fs.detach(),
                                                feat_teacher, unconditional)
                elif phase == 2:
                    # joint: both losses
                    loss_diff = _diffusion_loss(unet, scheduler, fs,
                                                feat_teacher, unconditional)
                elif phase == 3:
                    # student only: freeze diffusion
                    pass  # loss_diff stays 0

            # ── backward ─────────────────────────────────────────────────────
            opt_student.zero_grad()
            opt_diff.zero_grad()

            if phase == 1:
                # Only train diffusion
                loss_diff.backward()
                nn.utils.clip_grad_norm_(unet.parameters(), gc)
                opt_diff.step()
                total_loss = loss_diff
            elif phase == 2:
                # Joint
                total_loss = loss_bce + lambda_kd * loss_kd + lambda_diff * loss_diff
                total_loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), gc)
                nn.utils.clip_grad_norm_(unet.parameters(), gc)
                opt_student.step()
                opt_diff.step()
            else:  # phase 3
                total_loss = loss_bce + lambda_kd * loss_kd
                total_loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), gc)
                opt_student.step()

            ep_bce  += loss_bce.item()
            ep_kd   += loss_kd.item()
            ep_diff += loss_diff.item() if isinstance(loss_diff, torch.Tensor) else 0.0
            ep_tot  += total_loss.item()
            n_batches += 1

        sched_student.step()
        sched_diff.step()

        # ── validation ───────────────────────────────────────────────────────
        val_probs, val_labels, _ = _run_eval(student, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs, threshold=0.5)
        val_auc = val_metrics["macro_auc"]
        val_f1  = val_metrics.get("macro_f1", 0.0)

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(ckpt_path, student, epoch,
                            {"val_auc": val_auc},
                            {"variant_id": variant_id, "lambda_diff": lambda_diff,
                             "prediction_type": pred_type, "feat_layer": feat_layer})

        flag = " ★" if is_best else ""
        n  = n_batches
        line = (f"Ep {epoch:3d}/{total_ep} [P{phase}] | "
                f"bce={ep_bce/n:.4f} kd={ep_kd/n:.4f} diff={ep_diff/n:.4f} "
                f"tot={ep_tot/n:.4f} | val_auc={val_auc:.4f} val_f1={val_f1:.4f}{flag}")
        print(line)
        log_file.write(line + "\n"); log_file.flush()

        log_rows.append(dict(
            epoch=epoch, phase=phase,
            loss_bce=ep_bce/n, loss_kd=ep_kd/n, loss_diff=ep_diff/n,
            val_auc=val_auc, val_f1=val_f1, best=is_best,
        ))

    log_file.close()

    # ── test evaluation ───────────────────────────────────────────────────────
    print("\n[test evaluation]")
    load_checkpoint(ckpt_path, student, device)
    val_probs, val_labels, _ = _run_eval(student, val_loader, device)
    thresholds = find_best_thresholds(val_labels, val_probs)

    test_probs, test_labels, test_ids = _run_eval(student, test_loader, device)
    # AUC at 0.5 threshold
    m05        = compute_metrics(test_labels, test_probs, threshold=0.5)
    # F1 at tuned thresholds
    m_tune     = compute_metrics(test_labels, test_probs, threshold=thresholds)

    test_metrics = {**m05}
    test_metrics["macro_f1_tuned"] = round(m_tune.get("macro_f1", 0.0), 5)
    test_metrics["macro_f1_0_5"]   = round(m05.get("macro_f1", 0.0), 5)
    for c in CLASSES:
        test_metrics[f"f1_tuned_{c}"] = round(m_tune.get(f"f1_{c}", float("nan")), 5)

    t_min = (time.time() - t_start) / 60.0
    delta = test_metrics["macro_auc"] - LOCAL_F02T01_AUC
    flag  = "★ ABOVE BASELINE" if delta > 0.001 else ("~ TIE" if abs(delta) <= 0.0005 else "below baseline")

    print(f"\n{'='*65}")
    print(f"DiffKD result: {variant_id}")
    print(f"  test AUC  : {test_metrics['macro_auc']:.5f}  (ΔAUC={delta:+.5f})  {flag}")
    print(f"  F1_tuned  : {test_metrics['macro_f1_tuned']:.5f}")
    print(f"  val_best  : {best_auc:.5f}")
    print(f"  per-class : " + "  ".join(
        f"{c}={test_metrics.get(f'auc_{c}', 0):.4f}" for c in CLASSES))
    print(f"  time      : {t_min:.1f} min")
    print(f"{'='*65}\n")

    # ── save results ──────────────────────────────────────────────────────────
    extra = dict(
        variant_id=variant_id,
        lambda_diff=lambda_diff, lambda_kd=lambda_kd,
        temperature=temperature, w500=w500, w100=w100,
        feat_layer=feat_layer, prediction_type=pred_type,
        cond_type=cond_type, unconditional=unconditional,
        val_auc_best=best_auc,
        training_time_min=round(t_min, 1),
        local_f02t01_auc=LOCAL_F02T01_AUC,
        delta_auc=round(delta, 5),
        _meta=dict(seed=seed, epochs_total=total_ep),
    )
    _save_metrics(out_dir, "metrics_test.json", test_metrics, extra)

    # thresholds
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({c: float(thresholds[i]) for i, c in enumerate(CLASSES)}, f, indent=2)

    # train log CSV
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader(); w.writerows(log_rows)

    # summary append
    _append_summary(out_dir.parent, test_metrics, extra)

    print(f"Saved to: {out_dir}")


def _append_summary(out_dir: Path, metrics: dict, extra: dict) -> None:
    csv_path = out_dir / "summary.csv"
    row = dict(
        variant_id=extra["variant_id"],
        AUC_macro=metrics["macro_auc"],
        delta_AUC=extra["delta_auc"],
        F1_tuned=metrics["macro_f1_tuned"],
        val_best=extra["val_auc_best"],
        lambda_diff=extra["lambda_diff"],
        feat_layer=extra["feat_layer"],
        pred_type=extra["prediction_type"],
        unconditional=extra["unconditional"],
        time_min=extra["training_time_min"],
        **{f"auc_{c}": metrics.get(f"auc_{c}", "") for c in CLASSES},
    )
    rows: list[dict] = []
    if csv_path.exists():
        import csv
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get("variant_id") != row["variant_id"]]
    rows.append(row)
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiffKD trainer")

    # paths
    p.add_argument("--config",          default="dafd_mvkt/configs/adaptive_50hz.yaml")
    p.add_argument("--data_dir",        required=True)
    p.add_argument("--f02t01_ckpt",     required=True,
                   help="F02T01 best checkpoint (student init)")
    p.add_argument("--teacher_1l500_ckpt", required=True)
    p.add_argument("--teacher_1l100_ckpt", required=True)
    p.add_argument("--output_dir",      default="dafd_mvkt/outputs/diffkd")
    p.add_argument("--variant_id",      default="DKD00_dryrun")

    # KD (F02T01-identical defaults)
    p.add_argument("--weights",         default="0.60,0.40")
    p.add_argument("--temperature",     type=float, default=1.5)
    p.add_argument("--lambda_kd",       type=float, default=1.0)

    # Diffusion
    p.add_argument("--lambda_diff",     type=float, default=0.0,
                   help="diffusion loss weight (0.0 = dry-run)")
    p.add_argument("--diff_steps",      type=int,   default=100)
    p.add_argument("--pred_type",       default="v_prediction",
                   choices=["v_prediction", "epsilon"])
    p.add_argument("--feat_layer",      default="layer4",
                   choices=["layer3", "layer4"])
    p.add_argument("--cond_type",       default="concat",
                   choices=["concat", "attn"])
    p.add_argument("--unconditional",   action="store_true",
                   help="DKD03: no student conditioning")

    # U-Net architecture
    p.add_argument("--unet_base_ch",    type=int,   default=128)
    p.add_argument("--unet_mults",      default="1,2,4")

    # Phase schedule
    p.add_argument("--phase1_epochs",   type=int, default=0,
                   help="diffusion-only pretrain epochs")
    p.add_argument("--phase2_epochs",   type=int, default=0,
                   help="joint training epochs")
    p.add_argument("--phase3_epochs",   type=int, default=0,
                   help="student-only finetune epochs")

    # Standard
    p.add_argument("--epochs",          type=int, default=None,
                   help="override total epochs (sets phase2 only)")
    p.add_argument("--batch_size",      type=int, default=None)
    p.add_argument("--seed",            type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
