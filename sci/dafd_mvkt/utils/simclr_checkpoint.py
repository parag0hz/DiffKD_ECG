"""
SimCLR checkpoint utilities.

Functions:
  inspect_checkpoint(path)
      Print checkpoint structure (supports comper_repo ConvNeXt and dafd_mvkt ResNet1d).

  extract_encoder_state_dict(path)
      Handle multiple formats, return cleaned state_dict.

  convert_simclr_to_dafd(path, out_path)
      Extract and validate encoder weights, save in CLECG-compatible format.

  load_encoder_init(model, ckpt_path, strict=False)
      Load encoder weights with detailed reporting.

CLI:
  python dafd_mvkt/utils/simclr_checkpoint.py inspect --input <path>
  python dafd_mvkt/utils/simclr_checkpoint.py convert --input <in> --output <out>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn


# ── Known architecture signatures ────────────────────────────────────────────

_CONVNEXT_KEYS = frozenset(["downsample_layers", "stages"])
_RESNET_KEYS   = frozenset(["stem", "layer_blocks", "classifier", "proj_head",
                             "gap"])

# Prefixes to strip when loading into a target model
_STRIP_PREFIXES = [
    "module.", "encoder.", "backbone.", "model.", "net.", "f.",
    "query_encoder.", "query_enc.",
]


def _detect_architecture(keys: list[str]) -> str:
    top = {k.split(".")[0] for k in keys}
    if top & _CONVNEXT_KEYS:
        return "convnext"
    if top & _RESNET_KEYS:
        return "resnet1d"
    return "unknown"


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)}


def _clean_state_dict(raw_sd: dict) -> dict:
    """Strip known module-wrapper prefixes."""
    for prefix in _STRIP_PREFIXES:
        if all(k.startswith(prefix) for k in raw_sd):
            return _strip_prefix(raw_sd, prefix)
    return raw_sd


# ── Public API ────────────────────────────────────────────────────────────────

def inspect_checkpoint(path: str | Path) -> None:
    """Print checkpoint structure to stdout."""
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== Checkpoint: {path} ===")
    print(f"Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'raw state_dict'}")

    # Locate state dict
    sd = None
    if isinstance(ckpt, dict):
        for k in ["encoder_state_dict", "state_dict", "model_state_dict",
                  "backbone", "encoder", "query_enc"]:
            if k in ckpt:
                sd = ckpt[k]
                print(f"State dict key: '{k}'")
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
            print("Format: raw state_dict")
        # Print meta
        for k, v in ckpt.items():
            if not isinstance(v, (dict, torch.Tensor)):
                print(f"  meta[{k}]: {v}")
    else:
        sd = ckpt
        print("Format: raw state_dict (not wrapped)")

    if sd is None:
        print("  [WARN] Could not locate state_dict.")
        return

    keys = list(sd.keys())
    arch = _detect_architecture(keys)
    print(f"Architecture guess: {arch}")
    print(f"Total parameters: {len(keys)}")
    print("First 8 keys:")
    for k, v in list(sd.items())[:8]:
        print(f"  {k}: {tuple(v.shape)}")
    print("Last 3 keys:")
    for k, v in list(sd.items())[-3:]:
        print(f"  {k}: {tuple(v.shape)}")

    total_params = sum(v.numel() for v in sd.values())
    print(f"Total elements: {total_params:,}")

    if arch == "convnext":
        stem_key = [k for k in keys if "downsample_layers.0.0.weight" in k]
        if stem_key:
            shape = sd[stem_key[0]].shape
            print(f"\n[ConvNeXt] Stem shape: {tuple(shape)} "
                  f"→ in_channels={shape[1]}, out_channels={shape[0]}, kernel={shape[2]}")
    elif arch == "resnet1d":
        stem_keys = [k for k in keys if "stem" in k and "weight" in k]
        if stem_keys:
            shape = sd[stem_keys[0]].shape
            print(f"\n[ResNet1d] Stem conv shape: {tuple(shape)} "
                  f"→ in_channels={shape[1]}, out_channels={shape[0]}")


def extract_encoder_state_dict(path: str | Path) -> dict:
    """
    Load checkpoint and extract a cleaned encoder state_dict.

    Handles:
      - {"state_dict": ...}          dafd_mvkt CLECG/SimCLR format
      - {"encoder_state_dict": ...}  comper_repo lead-contrastive format
      - {"model_state_dict": ...}    generic
      - {"encoder": ...}             alternative
      - {"backbone": ...}            alternative
      - {"query_enc": ...}           CLECG full checkpoint
      - raw state_dict
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        for k in ["state_dict", "encoder_state_dict", "model_state_dict",
                  "encoder", "backbone", "query_enc"]:
            if k in ckpt:
                return _clean_state_dict(ckpt[k])
        # Check if it is a raw state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return _clean_state_dict(ckpt)
    elif hasattr(ckpt, "keys"):
        return _clean_state_dict(dict(ckpt))

    raise ValueError(f"Cannot extract encoder state_dict from {path}: "
                     f"top-level keys = {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")


def convert_simclr_to_dafd(
    path:     str | Path,
    out_path: str | Path,
    hz:       int = 100,
    lead:     str = "II",
) -> None:
    """
    Extract encoder state_dict and save in CLECG-compatible format.

    The output is directly loadable via --init_encoder_ckpt in
    train_student_hier.py / train_ta.py / train_temporal_ta.py.

    NOTE: comper_repo ConvNeXt checkpoints are NOT compatible with
    dafd_mvkt ResNet1d. This function will still save the state_dict
    (useful for inspection), but loading will fail with many missing keys.
    """
    path     = Path(path)
    out_path = Path(out_path)

    inspect_checkpoint(path)

    sd = extract_encoder_state_dict(path)
    arch = _detect_architecture(list(sd.keys()))

    if arch == "convnext":
        print(
            "\n[WARNING] Source checkpoint is ConvNeXt. It cannot be loaded "
            "into dafd_mvkt ResNet1d. Saving state_dict for reference only."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": sd, "hz": hz, "lead": lead,
         "source": str(path), "source_arch": arch},
        out_path,
    )
    print(f"\nSaved converted checkpoint: {out_path}")

    # Try to validate by loading into a ResNet1d
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from dafd_mvkt.models.resnet1d import ResNet1d
        model = ResNet1d(in_channels=1, num_classes=5)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"\nValidation (load into ResNet1d strict=False):")
        print(f"  Loaded    : {len(sd) - len(unexpected)} / {len(sd)} keys")
        print(f"  Missing   : {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (sample): {missing[:5]}")
        if arch == "convnext" and len(missing) > 0:
            print("  → Architecture mismatch confirmed: ConvNeXt → ResNet1d not loadable.")
    except ImportError:
        print("  (Validation skipped: dafd_mvkt not importable from this path)")


def load_encoder_init(
    model:     nn.Module,
    ckpt_path: str | Path,
    strict:    bool = False,
    device:    torch.device | None = None,
) -> tuple[list, list]:
    """
    Load encoder weights from a checkpoint into a model.

    Supports CLECG, SimCLR, and cleaned state_dict formats.
    Returns (missing_keys, unexpected_keys).

    Prints a summary of load results.
    """
    if device is None:
        device = next(model.parameters()).device

    sd = extract_encoder_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=strict)

    loaded = len(sd) - len(unexpected)
    print(f"  [load_encoder_init] loaded {loaded}/{len(sd)} keys from {ckpt_path}")
    if missing:
        print(f"    missing ({len(missing)}): {missing[:5]}"
              + ("..." if len(missing) > 5 else ""))
    if unexpected:
        print(f"    unexpected ({len(unexpected)}): {unexpected[:5]}"
              + ("..." if len(unexpected) > 5 else ""))

    return missing, unexpected


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli_inspect(args: argparse.Namespace) -> None:
    inspect_checkpoint(args.input)


def _cli_convert(args: argparse.Namespace) -> None:
    convert_simclr_to_dafd(
        args.input, args.output,
        hz=args.hz, lead=args.lead,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="SimCLR checkpoint utilities")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("inspect", help="Inspect checkpoint structure")
    pi.add_argument("--input", required=True)

    pc = sub.add_parser("convert", help="Convert checkpoint to dafd_mvkt format")
    pc.add_argument("--input",  required=True)
    pc.add_argument("--output", required=True)
    pc.add_argument("--hz",   type=int, default=100)
    pc.add_argument("--lead", default="II")

    args = p.parse_args()
    if args.cmd == "inspect":
        _cli_inspect(args)
    elif args.cmd == "convert":
        _cli_convert(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
