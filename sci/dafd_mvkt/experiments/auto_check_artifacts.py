"""
Preflight checks for the autonomous pipeline.

Verifies all required files exist, CLI args are present, and known bugs are patched.
Writes issues to outputs/auto/errors.log and exits non-zero on critical failure.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

AUTO_DIR = Path("dafd_mvkt/outputs/auto")
ERRORS_LOG = AUTO_DIR / "errors.log"

CRITICAL = "CRITICAL"
WARNING  = "WARNING"
OK       = "OK"


def log_error(msg: str, level: str = CRITICAL) -> None:
    AUTO_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERRORS_LOG, "a") as f:
        f.write(f"[{level}] {msg}\n")
    print(f"[{level}] {msg}", file=sys.stderr)


def check(cond: bool, ok_msg: str, fail_msg: str, level: str = CRITICAL) -> bool:
    if cond:
        print(f"  [OK] {ok_msg}")
        return True
    log_error(fail_msg, level)
    print(f"  [{level}] {fail_msg}", file=sys.stderr)
    return level != CRITICAL


def check_file(path: str, desc: str) -> bool:
    return check(Path(path).exists(), f"{desc}: {path}", f"Missing {desc}: {path}")


def check_arg(script: str, arg: str) -> bool:
    """Check if a Python script accepts --arg."""
    try:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True, text=True, timeout=10
        )
        return arg in result.stdout
    except Exception:
        return False


def patch_yaml(path: str, key: str, value: str) -> None:
    """Append key: value to a YAML file if not already present."""
    content = Path(path).read_text()
    if key not in content:
        with open(path, "a") as f:
            f.write(f"  {key}: {value}\n")
        print(f"  [PATCH] Added {key}: {value} to {path}")


def main(args: argparse.Namespace) -> int:
    AUTO_DIR.mkdir(parents=True, exist_ok=True)
    # Clear errors log for fresh run
    ERRORS_LOG.write_text("")

    failures = 0

    print("\n=== Preflight Check ===")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[Data]")
    data_dir = Path(args.data_dir)
    if not check(data_dir.exists(), f"DATA_DIR: {data_dir}", f"DATA_DIR not found: {data_dir}"):
        failures += 1
    else:
        for f in ["ptbxl_database.csv", "scp_statements.csv"]:
            if not check_file(str(data_dir / f), f):
                failures += 1
        records_ok = (data_dir / "records500").exists() or (data_dir / "records100").exists()
        if not check(records_ok, "records500 or records100", "No records500/records100 directory"):
            failures += 1

    # ── Teacher checkpoint ────────────────────────────────────────────────────
    print("\n[Teacher checkpoint]")
    teacher_ckpt = "dafd_mvkt/outputs/teacher_best.pt"
    if not check_file(teacher_ckpt, "teacher_best.pt"):
        failures += 1
        log_error(
            f"Train teacher first:\n"
            f"  python dafd_mvkt/train_teacher.py "
            f"--config dafd_mvkt/configs/teacher_500hz.yaml "
            f"--data_dir {args.data_dir}",
            CRITICAL,
        )

    # ── Python scripts ────────────────────────────────────────────────────────
    print("\n[Python scripts]")
    scripts = [
        "dafd_mvkt/pretrain_clecg.py",
        "dafd_mvkt/train_ta.py",
        "dafd_mvkt/train_temporal_ta.py",
        "dafd_mvkt/train_student_hier.py",
        "dafd_mvkt/evaluate.py",
        "dafd_mvkt/experiments/aggregate_results.py",
        "dafd_mvkt/experiments/make_paper_tables.py",
    ]
    for s in scripts:
        if not check_file(s, s):
            failures += 1

    # ── CLI arg checks ────────────────────────────────────────────────────────
    print("\n[CLI args]")
    for script, arg in [
        ("dafd_mvkt/train_student_hier.py", "--init_encoder_ckpt"),
        ("dafd_mvkt/train_ta.py",           "--init_encoder_ckpt"),
        ("dafd_mvkt/train_ta.py",           "--output_dir"),
        ("dafd_mvkt/train_ta.py",           "--seed"),
        ("dafd_mvkt/train_temporal_ta.py",  "--init_encoder_ckpt"),
        ("dafd_mvkt/train_student_hier.py", "--lr"),
        ("dafd_mvkt/train_student_hier.py", "--alpha_ta"),
    ]:
        if not check(check_arg(script, arg), f"{script} {arg}", f"{script} missing {arg}", WARNING):
            pass  # warning only

    # ── eval_input_key in progressive configs ─────────────────────────────────
    print("\n[Config eval_input_key]")
    for cfg_path in [
        "dafd_mvkt/configs/ta_ii_100hz_progressive.yaml",
        "dafd_mvkt/configs/ta_i_100hz_progressive.yaml",
    ]:
        if Path(cfg_path).exists():
            content = Path(cfg_path).read_text()
            if "eval_input_key" not in content:
                log_error(f"{cfg_path} missing eval_input_key", WARNING)
                patch_yaml(cfg_path, "eval_input_key", "student_x")
            else:
                print(f"  [OK] {cfg_path} has eval_input_key")
        else:
            check(False, cfg_path, f"Config not found: {cfg_path}", WARNING)

    # ── evaluate.py eval_input_key support ────────────────────────────────────
    eval_content = Path("dafd_mvkt/evaluate.py").read_text() if Path("dafd_mvkt/evaluate.py").exists() else ""
    check("eval_input_key" in eval_content,
          "evaluate.py supports eval_input_key",
          "evaluate.py missing eval_input_key support", WARNING)

    # ── Output dirs ───────────────────────────────────────────────────────────
    print("\n[Output directories]")
    for d in ["dafd_mvkt/outputs/auto", "dafd_mvkt/outputs/clecg",
              "dafd_mvkt/outputs/tuning", "dafd_mvkt/outputs/final",
              "dafd_mvkt/outputs/summary", "dafd_mvkt/outputs/paper_tables"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    if failures > 0:
        print(f"PREFLIGHT FAILED: {failures} critical issue(s). See {ERRORS_LOG}")
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="comper_repo/ptb_xl")
    sys.exit(main(p.parse_args()))
