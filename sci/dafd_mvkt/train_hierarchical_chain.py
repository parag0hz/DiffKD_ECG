"""
Hierarchical chain KD trainer.

For chain Hxx = [t1_view, t2_view, t3_view], trains:
  t1 → t2   (stage 2)
  t2 → t3   (stage 3)
  t3 → S    (stage 4: S = 1L50 student)

Usage:
    python dafd_mvkt/train_hierarchical_chain.py \\
        --config   dafd_mvkt/configs/hierarchical_chains_6.yaml \\
        --data_dir comper_repo/ptb_xl \\
        --chain_id H02 \\
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
from utils.seed import set_seed


# ── reference constants ───────────────────────────────────────────────────────

BCE_AUC  = 0.8061;  BCE_F1  = 0.5740
PROG_AUC = 0.8396;  PROG_F1 = 0.6176
C14_AUC  = 0.8425;  C14_F1  = 0.6243
MVKT_AUC = 0.843;   MVKT_F1 = 0.626


def _resolve_t1_ckpt(t1_view: str, chain_cfg: dict, seed: int) -> str:
    """Return path to pre-trained t1 checkpoint. Raises if missing."""
    t1_map = chain_cfg.get("t1_checkpoints", {})
    ckpt = t1_map.get(t1_view)
    if not ckpt:
        raise ValueError(f"No t1 checkpoint defined for view {t1_view} in config")
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(
            f"t1 checkpoint not found: {p}\n"
            f"Run: DATA_DIR=comper_repo/ptb_xl SEED={seed} LEAD=II "
            f"bash dafd_mvkt/scripts/prepare_3teacher_bank.sh"
        )
    return str(p)


def _resolve_simclr_ckpt(view: str, chain_cfg: dict, seed: int) -> str | None:
    """Return SimCLR encoder path for target view if it exists, else None."""
    simclr_map = chain_cfg.get("simclr_checkpoints", {})
    tmpl = simclr_map.get(view)
    if not tmpl:
        return None
    path = Path(tmpl.replace("{seed}", str(seed)))
    if path.exists():
        return str(path)
    print(f"[WARN] SimCLR ckpt not found for view {view}: {path}  → random init")
    return None


def run_chain(
    chain_id:           str,
    chain_cfg:          dict,
    data_dir:           str,
    lead:               str,
    seed:               int,
    output_dir:         Path,
    reuse_intermediate: bool = False,
    debug:              bool = False,
) -> dict:
    """
    Run all stages of one hierarchical chain.

    Returns final student metrics dict.
    """
    chains     = chain_cfg["chains"]
    if chain_id not in chains:
        raise ValueError(f"Unknown chain_id={chain_id}. Available: {list(chains)}")

    t1_view, t2_view, t3_view = chains[chain_id]
    student_view = "1L50"

    print(f"\n{'#'*64}")
    print(f"  Chain {chain_id}: {t1_view} → {t2_view} → {t3_view} → S({student_view})")
    print(f"  seed={seed}  reuse_intermediate={reuse_intermediate}")
    print(f"{'#'*64}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── stage naming ──────────────────────────────────────────────────────────
    s2_name = f"{chain_id}_stage2_{t2_view}_from_{t1_view}_seed{seed}"
    s3_name = f"{chain_id}_stage3_{t3_view}_from_{t2_view}_seed{seed}"

    student_name = (
        f"student_II_50hz_hier_{chain_id}_"
        f"{t1_view}_{t2_view}_{t3_view}_simclr_seed{seed}"
    ).replace(" ", "")

    # paths
    s2_ckpt      = output_dir / f"{s2_name}_best.pt"
    s3_ckpt      = output_dir / f"{s3_name}_best.pt"
    student_ckpt = output_dir / f"{student_name}_best.pt"
    student_metrics_path = output_dir / f"metrics_test_{student_name}.json"

    # ── resolve t1 checkpoint ─────────────────────────────────────────────────
    t1_ckpt = _resolve_t1_ckpt(t1_view, chain_cfg, seed)
    print(f"\n[Stage 1] t1={t1_view}  ckpt: {t1_ckpt}")

    # ── stage 2: t1 → t2 ─────────────────────────────────────────────────────
    s2_metrics_path = output_dir / f"metrics_test_{s2_name}.json"

    if s2_metrics_path.exists() and s2_ckpt.exists():
        print(f"\n[Stage 2] SKIP (exists): {s2_metrics_path}")
        t2_ckpt = str(s2_ckpt)
    elif reuse_intermediate and t2_view in chain_cfg.get("t1_checkpoints", {}):
        t2_ckpt = chain_cfg["t1_checkpoints"][t2_view]
        print(f"\n[Stage 2] REUSE existing t2 ckpt: {t2_ckpt}")
    else:
        print(f"\n[Stage 2] Training: {t1_view} → {t2_view}")
        t2_init = _resolve_simclr_ckpt(t2_view, chain_cfg, seed)
        train_kd_view(
            chain_cfg=chain_cfg,
            data_dir=data_dir,
            lead=lead,
            teacher_view=t1_view,
            target_view=t2_view,
            teacher_ckpt=t1_ckpt,
            output_name=s2_name,
            output_dir=output_dir,
            target_init_ckpt=t2_init,
            lambda_kd=chain_cfg["training"]["lambda_kd"],
            temperature=chain_cfg["training"]["temperature"],
            seed=seed,
            debug=debug,
        )
        t2_ckpt = str(s2_ckpt)

    # ── stage 3: t2 → t3 ─────────────────────────────────────────────────────
    s3_metrics_path = output_dir / f"metrics_test_{s3_name}.json"

    if s3_metrics_path.exists() and s3_ckpt.exists():
        print(f"\n[Stage 3] SKIP (exists): {s3_metrics_path}")
        t3_ckpt = str(s3_ckpt)
    elif reuse_intermediate and t3_view in chain_cfg.get("t1_checkpoints", {}):
        t3_ckpt = chain_cfg["t1_checkpoints"][t3_view]
        print(f"\n[Stage 3] REUSE existing t3 ckpt: {t3_ckpt}")
    else:
        print(f"\n[Stage 3] Training: {t2_view} → {t3_view}")
        t3_init = _resolve_simclr_ckpt(t3_view, chain_cfg, seed)
        train_kd_view(
            chain_cfg=chain_cfg,
            data_dir=data_dir,
            lead=lead,
            teacher_view=t2_view,
            target_view=t3_view,
            teacher_ckpt=t2_ckpt,
            output_name=s3_name,
            output_dir=output_dir,
            target_init_ckpt=t3_init,
            lambda_kd=chain_cfg["training"]["lambda_kd"],
            temperature=chain_cfg["training"]["temperature"],
            seed=seed,
            debug=debug,
        )
        t3_ckpt = str(s3_ckpt)

    # ── stage 4: t3 → student(1L50) ──────────────────────────────────────────
    if student_metrics_path.exists() and student_ckpt.exists():
        print(f"\n[Stage 4] SKIP (exists): {student_metrics_path}")
        with open(student_metrics_path) as f:
            return json.load(f)

    print(f"\n[Stage 4] Training final student: {t3_view} → {student_view}")
    s4_init = _resolve_simclr_ckpt(student_view, chain_cfg, seed)

    metrics = train_kd_view(
        chain_cfg=chain_cfg,
        data_dir=data_dir,
        lead=lead,
        teacher_view=t3_view,
        target_view=student_view,
        teacher_ckpt=t3_ckpt,
        output_name=student_name,
        output_dir=output_dir,
        target_init_ckpt=s4_init,
        lambda_kd=chain_cfg["training"]["lambda_kd"],
        temperature=chain_cfg["training"]["temperature"],
        seed=seed,
        debug=debug,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    auc = metrics["macro_auc"]
    f1t = metrics["macro_f1_tuned"]
    print(f"\n{'='*60}")
    print(f"Chain {chain_id}: {t1_view}→{t2_view}→{t3_view}→S")
    print(f"  AUC       : {auc:.4f}  (MVKT={MVKT_AUC})")
    print(f"  F1_tuned  : {f1t:.4f}  (MVKT={MVKT_F1})")
    print(f"  vs BCE    : {auc - BCE_AUC:+.4f}  F1: {f1t - BCE_F1:+.4f}")
    print(f"  vs ProgSCL: {auc - PROG_AUC:+.4f}  F1: {f1t - PROG_F1:+.4f}")
    print(f"  vs C14    : {auc - C14_AUC:+.4f}  F1: {f1t - C14_F1:+.4f}")
    print(f"  Beats MVKT AUC: {'YES ✓' if auc > MVKT_AUC else 'no'}")
    print(f"  Beats MVKT F1 : {'YES ✓' if f1t > MVKT_F1 else 'no'}")
    print(f"{'='*60}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="dafd_mvkt/configs/hierarchical_chains_6.yaml")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--lead",     default="II")
    p.add_argument("--chain_id", required=True, choices=["H01","H02","H03","H04","H05","H06"])
    p.add_argument("--reuse_intermediate", action="store_true")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--debug",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        chain_cfg = yaml.safe_load(f)

    run_chain(
        chain_id=args.chain_id,
        chain_cfg=chain_cfg,
        data_dir=args.data_dir,
        lead=args.lead,
        seed=args.seed,
        output_dir=Path("dafd_mvkt/outputs/hierchain"),
        reuse_intermediate=args.reuse_intermediate,
        debug=args.debug,
    )
