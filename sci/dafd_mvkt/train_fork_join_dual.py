"""
Fork-join dual distillation runner.

Given fork Fxx = {t1, t2, t3}:
  Stage A: resolve t1 checkpoint
  Stage B: train t2 from t1  (train_kd_view)
  Stage C: train t3 from t1  (train_kd_view)
  Stage D: train student from t2+t3  (train_student_dual_teacher)

Usage:
    python dafd_mvkt/train_fork_join_dual.py \\
        --config   dafd_mvkt/configs/fork_join_dual_6.yaml \\
        --data_dir comper_repo/ptb_xl \\
        --fork_id  F02 \\
        --seed     0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from train_kd_view import train_kd_view, VIEW_DEFS
from train_student_dual_teacher import train_student_dual_teacher
from utils.seed import set_seed

BCE_AUC  = 0.8061;  BCE_F1  = 0.5740
PROG_AUC = 0.8396;  PROG_F1 = 0.6176
C14_AUC  = 0.8425;  C14_F1  = 0.6243
MVKT_AUC = 0.843;   MVKT_F1 = 0.626


def _resolve_t1_ckpt(t1_view: str, fork_cfg: dict, seed: int) -> str:
    t1_map = fork_cfg.get("t1_checkpoints", {})
    ckpt   = t1_map.get(t1_view)
    if not ckpt:
        raise ValueError(f"No t1 checkpoint for view {t1_view}")
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(
            f"t1 checkpoint not found: {p}\n"
            f"Run: DATA_DIR=comper_repo/ptb_xl SEED={seed} LEAD=II "
            f"bash dafd_mvkt/scripts/prepare_3teacher_bank.sh"
        )
    return str(p)


def _resolve_simclr(view: str, fork_cfg: dict, seed: int) -> str | None:
    tmpl = fork_cfg.get("simclr_checkpoints", {}).get(view)
    if not tmpl:
        return None
    p = Path(tmpl.replace("{seed}", str(seed)))
    if p.exists():
        return str(p)
    return None


def run_fork_join(
    fork_id:        str,
    fork_cfg:       dict,
    data_dir:       str,
    lead:           str,
    seed:           int,
    output_dir:     Path,
    reuse_branches: bool = False,
    debug:          bool = False,
) -> dict:
    forks = fork_cfg["fork_join"]
    if fork_id not in forks:
        raise ValueError(f"Unknown fork_id={fork_id}. Available: {list(forks)}")

    spec    = forks[fork_id]
    t1_view = spec["t1"]
    t2_view = spec["t2"]
    t3_view = spec["t3"]

    print(f"\n{'#'*64}")
    print(f"  Fork {fork_id}: {t1_view} → {{{t2_view}, {t3_view}}} → S(1L50)")
    print(f"  seed={seed}  reuse_branches={reuse_branches}")
    print(f"{'#'*64}")

    output_dir.mkdir(parents=True, exist_ok=True)

    b2_name = f"{fork_id}_branch2_{t2_view}_from_{t1_view}_seed{seed}"
    b3_name = f"{fork_id}_branch3_{t3_view}_from_{t1_view}_seed{seed}"
    s_name  = (f"student_II_50hz_forkjoin_{fork_id}_"
               f"{t1_view}_to_{t2_view}_{t3_view}_simclr_seed{seed}")

    b2_ckpt      = output_dir / f"{b2_name}_best.pt"
    b3_ckpt      = output_dir / f"{b3_name}_best.pt"
    s_ckpt       = output_dir / f"{s_name}_best.pt"
    s_metrics_path = output_dir / f"metrics_test_{s_name}.json"

    # ── Stage A: resolve t1 ───────────────────────────────────────────────────
    t1_ckpt = _resolve_t1_ckpt(t1_view, fork_cfg, seed)
    print(f"\n[Stage A] t1={t1_view}  ckpt: {t1_ckpt}")

    # ── Stage B: t1 → t2 ──────────────────────────────────────────────────────
    b2_metrics_path = output_dir / f"metrics_test_{b2_name}.json"
    if b2_metrics_path.exists() and b2_ckpt.exists():
        print(f"\n[Stage B] SKIP (exists): {b2_name}")
    elif reuse_branches and t2_view in fork_cfg.get("t1_checkpoints", {}):
        print(f"\n[Stage B] REUSE global ckpt for {t2_view}")
        import shutil
        shutil.copy(fork_cfg["t1_checkpoints"][t2_view], str(b2_ckpt))
    else:
        print(f"\n[Stage B] Training: {t1_view} → {t2_view}")
        t2_init = _resolve_simclr(t2_view, fork_cfg, seed)
        train_kd_view(
            chain_cfg=fork_cfg,
            data_dir=data_dir,
            lead=lead,
            teacher_view=t1_view,
            target_view=t2_view,
            teacher_ckpt=t1_ckpt,
            output_name=b2_name,
            output_dir=output_dir,
            target_init_ckpt=t2_init,
            lambda_kd=fork_cfg["training"]["lambda_kd"],
            temperature=fork_cfg["training"]["temperature"],
            seed=seed,
            debug=debug,
        )

    # ── Stage C: t1 → t3 ──────────────────────────────────────────────────────
    b3_metrics_path = output_dir / f"metrics_test_{b3_name}.json"
    if b3_metrics_path.exists() and b3_ckpt.exists():
        print(f"\n[Stage C] SKIP (exists): {b3_name}")
    elif reuse_branches and t3_view in fork_cfg.get("t1_checkpoints", {}):
        print(f"\n[Stage C] REUSE global ckpt for {t3_view}")
        import shutil
        shutil.copy(fork_cfg["t1_checkpoints"][t3_view], str(b3_ckpt))
    else:
        print(f"\n[Stage C] Training: {t1_view} → {t3_view}")
        t3_init = _resolve_simclr(t3_view, fork_cfg, seed)
        train_kd_view(
            chain_cfg=fork_cfg,
            data_dir=data_dir,
            lead=lead,
            teacher_view=t1_view,
            target_view=t3_view,
            teacher_ckpt=t1_ckpt,
            output_name=b3_name,
            output_dir=output_dir,
            target_init_ckpt=t3_init,
            lambda_kd=fork_cfg["training"]["lambda_kd"],
            temperature=fork_cfg["training"]["temperature"],
            seed=seed,
            debug=debug,
        )

    # ── Stage D: t2+t3 → student ──────────────────────────────────────────────
    if s_metrics_path.exists() and s_ckpt.exists():
        print(f"\n[Stage D] SKIP (exists): {s_name}")
        with open(s_metrics_path) as f:
            return json.load(f)

    print(f"\n[Stage D] Training final student: {t2_view}+{t3_view} → 1L50")
    s_init = _resolve_simclr("1L50", fork_cfg, seed)

    metrics = train_student_dual_teacher(
        fork_cfg=fork_cfg,
        data_dir=data_dir,
        lead=lead,
        teacher2_view=t2_view,
        teacher3_view=t3_view,
        teacher2_ckpt=str(b2_ckpt),
        teacher3_ckpt=str(b3_ckpt),
        teacher_weights="0.5,0.5",
        output_name=s_name,
        output_dir=output_dir,
        student_init_ckpt=s_init,
        lambda_kd=fork_cfg["training"]["lambda_kd"],
        temperature=fork_cfg["training"]["temperature"],
        seed=seed,
        debug=debug,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    auc = metrics["macro_auc"]
    f1t = metrics["macro_f1_tuned"]
    print(f"\n{'='*60}")
    print(f"Fork {fork_id}: {t1_view}→{{{t2_view},{t3_view}}}→S")
    print(f"  AUC       : {auc:.4f}  (MVKT={MVKT_AUC})")
    print(f"  F1_tuned  : {f1t:.4f}  (MVKT={MVKT_F1})")
    print(f"  vs BCE    : {auc-BCE_AUC:+.4f}  F1: {f1t-BCE_F1:+.4f}")
    print(f"  vs Prog   : {auc-PROG_AUC:+.4f}  F1: {f1t-PROG_F1:+.4f}")
    print(f"  vs C14    : {auc-C14_AUC:+.4f}  F1: {f1t-C14_F1:+.4f}")
    print(f"  Beats MVKT AUC: {'YES ✓' if auc > MVKT_AUC else 'no'}")
    print(f"  Beats MVKT F1 : {'YES ✓' if f1t > MVKT_F1  else 'no'}")
    print(f"{'='*60}")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="dafd_mvkt/configs/fork_join_dual_6.yaml")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--lead",     default="II")
    p.add_argument("--fork_id",  required=True,
                   choices=["F01","F02","F03","F04","F05","F06"])
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--reuse_branches", action="store_true")
    p.add_argument("--debug",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        fork_cfg = yaml.safe_load(f)

    run_fork_join(
        fork_id=args.fork_id,
        fork_cfg=fork_cfg,
        data_dir=args.data_dir,
        lead=args.lead,
        seed=args.seed,
        output_dir=Path("dafd_mvkt/outputs/forkjoin"),
        reuse_branches=args.reuse_branches,
        debug=args.debug,
    )
