# SimCLR Repository Inspection Report

Generated: 2026-05-20

## 1. Discovered Files in comper_repo

| File | Description |
|---|---|
| `comper_repo/training/train_simclr.py` | 12-lead SimCLR pretraining (ConvNeXtContrastive) |
| `comper_repo/training/train_lead_contrastive.py` | Lead-masking contrastive pretraining |
| `comper_repo/training/loss.py` | `SimCLRLoss`, `nt_xent_loss()` |
| `comper_repo/models/model_convnext.py` | `ConvNeXt1d_Encoder` |
| `comper_repo/models/model_convnext_transformer.py` | `ConvNeXtTransformerUNet` |

Existing checkpoints:
- `experiments_lead_cl/1v1/encoder_pretrained.pth` (epoch 96, loss 0.168)
- `experiments_lead_cl/3v1/encoder_pretrained.pth`
- `experiments_lead_cl/12v1/encoder_pretrained.pth`
- `experiments_lead_cl/randvrand/encoder_pretrained.pth`
- `experiments_lead_cl_fixed/` (same variants, fixed augmentation)

## 2. Entrypoint Training Script

`comper_repo/training/train_lead_contrastive.py`
- Supports strategies: `1v1`, `3v1`, `12v1`, `randvrand`, `einthoven_12v1`
- `1v1`: two single-lead views (different random leads each)

## 3. Dataset Class

`LeadContrastiveDataset` in `train_lead_contrastive.py`
- Data: PTB-XL 100 Hz, stored as (N, 1000, 12)
- For `1v1`: selects 2 different random leads → each view is (1, L)
- Supports folds 1-8 for training

## 4. Encoder Architecture

**Class:** `ConvNeXt1d_Encoder` in `comper_repo/models/model_convnext.py`

```
Input:  (B, 12, 1000)
Stem:   Conv1d(12→96, k=4, s=4) + LayerNorm  → (B, 96, 250)
Stage1: 3×ConvNeXtBlock(96)                  → (B, 96, 250)
Down1:  Conv1d(96→192, k=2, s=2)             → (B, 192, 125)
Stage2: 3×ConvNeXtBlock(192)                 → (B, 192, 125)
Down2:  Conv1d(192→384, k=2, s=2)            → (B, 384, 62)
Stage3: 9×ConvNeXtBlock(384)                 → (B, 384, 62)
Down3:  Conv1d(384→768, k=2, s=2)            → (B, 768, 31)
Stage4: 3×ConvNeXtBlock(768)                 → (B, 768, 31)
Output: (B, 768, 31)
```

Checkpoint key: `encoder_state_dict`
Parameter names: `downsample_layers.*`, `stages.*`

## 5. Augmentation Methods

From `ECGAugmentor` in `train_lead_contrastive.py`:
- Gaussian noise: `std = signal.std() × 0.05`, p=0.8
- Baseline wander: low-freq sine (0.05-0.5 Hz), p=0.5
- Amplitude scaling: uniform(0.8, 1.2), p=0.5
- Cardiac axis rotation (for sparse view): θ~U(-45°, 90°)

## 6. Loss Function

`nt_xent_loss(z1, z2, temperature=0.1)` in `train_lead_contrastive.py`:
- Standard NT-Xent (symmetric)
- Runs in float32 (autocast disabled)
- Positive pairs: (z1[i], z2[i]) and (z2[i], z1[i])
- Negatives: all in-batch samples

## 7. Expected Input Shape

`(B, 12, 1000)` — 12 leads, 1000 time samples (100 Hz × 10 s)

## 8. Sampling Rate Assumptions

100 Hz assumed (1000 samples = 10 seconds).

## 9. Checkpoint Saving Format

```python
torch.save({
    "epoch": int,
    "loss": float,
    "encoder_state_dict": OrderedDict,  # ConvNeXt weights only
    "pos_strategy": str,
    "temperature": float,
}, "encoder_pretrained.pth")
```

## 10. Single-Lead Support

YES (partially): `1v1` strategy randomly selects one lead per view, but the
encoder still expects 12-channel input and zero-fills other leads. It does NOT
natively support true single-lead (1-channel) encoder training.

## 11. PTB-XL Support

YES, but requires `data_100hz/x_train.npy` format (pre-processed numpy arrays).

## 12. Compatibility with dafd_mvkt ResNet1d

**NOT COMPATIBLE. Direct weight transfer is impossible.**

| Aspect | comper_repo | dafd_mvkt |
|---|---|---|
| Architecture | `ConvNeXt1d_Encoder` | `ResNet1d` |
| Input channels | 12 (zero-fills non-active) | 1 (true single-lead) |
| Feat dim | 768 | 512 |
| Parameter names | `downsample_layers.*`, `stages.*` | `stem.*`, `layer_blocks.*` |
| Key `[0,0]` shape | `[96, 12, 4]` | `[64, 1, 15]` |

Confirmed by inspecting `experiments_lead_cl/1v1/encoder_pretrained.pth`:
- 178 parameter tensors
- First key: `downsample_layers.0.0.weight: [96, 12, 4]`
- Top-level modules: `{downsample_layers, stages}` (ConvNeXt, NOT ResNet)

## 13. Verdict

**comper_repo SimCLR is NOT directly reusable for dafd_mvkt.**

Loss function (NT-Xent) and augmentation concepts are reusable.
A new `pretrain_simclr.py` implemented in dafd_mvkt uses the same `ResNet1d`
backbone as downstream classification.

**Implementation path:** `dafd_mvkt/pretrain_simclr.py` (new, ResNet1d-based)
Checkpoints saved as `{"state_dict": ..., "hz": ..., "lead": ...}` — identical
to `pretrain_clecg.py` format, directly loadable via `--init_encoder_ckpt`.
