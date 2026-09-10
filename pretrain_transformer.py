#!/usr/bin/env python3
"""
Standalone script version of pretrain_transformer.ipynb, for direct SLURM
execution (python pretrain_transformer.py) instead of jupyter nbconvert
--execute. print() output now goes straight to real stdout, so it
actually lands in TRA.out -- nbconvert's execute+capture-to-notebook-JSON
flow does not stream cell output live, and loses it entirely if the
kernel is killed (e.g. OOM) before the conversion finishes writing the
output notebook.

Run `pip install crunch-cli torch` and
`crunch setup-notebook datacrunch-2 0JiCmmP21Ca88X8TApRuDHMH --size small`
as plain shell commands BEFORE this script -- see euler_transformer.sh.
IPython magics (%pip, !shell) only work inside a notebook/IPython kernel,
not in a plain python script.
"""

import resource


def report_memory(label: str) -> None:
    """Peak resident-set size so far, from the OS itself -- no extra
    package needed. flush=True matters here specifically: an OOM-kill is a
    SIGKILL from outside the process, so any buffered-but-unflushed print()
    output can be lost entirely -- we want every checkpoint to actually hit
    the log file the instant it's called, not sit in a buffer."""
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB on Linux -> GB
    print(f"[memory] peak RSS so far: {peak_gb:.2f} GB  ({label})", flush=True)


report_memory("script start (pip install / crunch setup-notebook already ran as separate bash steps -- see euler_transformer.sh)")


import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import crunch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: no GPU detected -- pretraining here will be slow. Check your CUDA setup.")

report_memory("after imports")

crunch_tools = crunch.load_notebook()


RANDOM_STATE = 0
ID_COLUMNS = ["id", "moon"]
CHECKPOINT_PATH = "transformer_checkpoint.pt"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in ID_COLUMNS and c != "target"]


def spearman(y_true, y_pred) -> float:
    corr, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


class FTTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, n_bins: int = 7, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.value_embedding = nn.Embedding(n_bins, d_model)
        self.feature_id_embedding = nn.Embedding(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_bins: torch.Tensor) -> torch.Tensor:
        # x_bins: (batch, n_features) int64 in [0, n_bins)
        batch_size = x_bins.shape[0]
        feature_ids = torch.arange(self.n_features, device=x_bins.device).unsqueeze(0)
        tokens = self.value_embedding(x_bins) + self.feature_id_embedding(feature_ids)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out).squeeze(-1)


def build_model(config: dict) -> FTTransformerRegressor:
    return FTTransformerRegressor(
        n_features=config["n_features"],
        n_bins=config.get("n_bins", 7),
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 3),
        dropout=config.get("dropout", 0.1),
    )


def save_checkpoint(path, model, config, feature_columns, target_mean, target_std):
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "feature_columns": feature_columns,
        "target_mean": target_mean,
        "target_std": target_std,
    }, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint


X_train, y_train, X_test = crunch_tools.load_data()

# MEMORY FIX 1: Immediately drop the unused test set to free up system RAM
del X_test
gc.collect()

feature_columns = get_feature_columns(X_train)
print(f"{len(feature_columns)} features, {len(X_train):,} rows")
report_memory("after load_data() and freeing test set")

# Features are already quantized into 7 bins (0-6) -- int8 instead of the
# default float64 cuts memory roughly 8x. This is the single biggest lever
# on the ~40GB RAM footprint you mentioned. NOTE: right after this line,
# the float64 X_train columns AND the new int8 copy briefly coexist --
# that's the single highest-memory moment in this notebook, which is why
# there's a checkpoint immediately before and after it.
X_bins_all = X_train[feature_columns].to_numpy().astype(np.int8)
report_memory("after building int8 feature matrix (float64 X_train copy still alive here)")

# Free the float64 feature columns now that we have the int8 copy -- keep
# only id/moon, which the holdout split below still needs. Don't rely on
# X_train eventually going out of scope; drop it explicitly.
X_train = X_train[["id", "moon"]]
gc.collect()
report_memory("after freeing float64 feature columns")

print(f"Feature matrix memory: {X_bins_all.nbytes / 1e9:.2f} GB (int8)")


moons = np.sort(X_train["moon"].unique())
holdout_moons = moons[-50:]
is_holdout = X_train["moon"].isin(holdout_moons).to_numpy()

target = y_train["target"].to_numpy()
target_mean, target_std = float(target[~is_holdout].mean()), float(target[~is_holdout].std())
target_scaled = (target - target_mean) / target_std

X_fit_bins, y_fit = X_bins_all[~is_holdout], target_scaled[~is_holdout]
X_val_bins, y_val_scaled = X_bins_all[is_holdout], target_scaled[is_holdout]
y_val_raw = target[is_holdout]

# Boolean indexing on a numpy array copies rather than views, so X_bins_all
# briefly coexists with the new X_fit_bins/X_val_bins copies -- free it
# once the split is made, same reasoning as above.
del X_bins_all
gc.collect()
report_memory("after holdout split")

print(f"train rows: {len(y_fit):,}   holdout rows: {len(y_val_scaled):,}")


PRETRAIN_CONFIG = {
    "n_bins": 7,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
}
PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 3e-4

# MEMORY FIX 2: Reduce batch size and use gradient accumulation
# 512 * 16 = 8192 (maintaining your original effective batch size)
PRETRAIN_BATCH_SIZE = 512//2  
ACCUMULATION_STEPS = 16//2    
WEIGHT_DECAY = 1e-5


# Manual batching over in-memory tensors -- no DataLoader/multiprocessing.
# Memory: X_fit_bins/X_val_bins are int8 (1 byte/value). nn.Embedding
# requires int64 indices, but casting the WHOLE array to .long() up front
# turns that back into an 8-bytes/value tensor -- undoing the int8 cast
# and the entire point of it. Instead we keep the full tensors as int8 and
# only cast .long() on each small batch slice, right before it's used.
X_fit_bins_t = torch.from_numpy(X_fit_bins)   # stays int8
y_fit_tensor = torch.from_numpy(y_fit).float()
X_val_bins_t = torch.from_numpy(X_val_bins)   # stays int8
n_train = len(y_fit_tensor)

model = build_model({**PRETRAIN_CONFIG, "n_features": len(feature_columns)}).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)
loss_fn = nn.MSELoss()

# MEMORY FIX 3: Initialize scaler for PyTorch Mixed Precision (AMP)
use_amp = (DEVICE == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def predict_in_batches(model, X_bins_t, batch_size=PRETRAIN_BATCH_SIZE):
    """Batched inference -- same int8-stays-int8-until-the-last-moment
    reasoning as training: never materialize a whole-dataset int64 copy."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_bins_t), batch_size):
            xb = X_bins_t[start:start + batch_size].long().to(DEVICE)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


best_val_spearman = -1.0
best_state = None

for epoch in range(PRETRAIN_EPOCHS):
    model.train()
    total_loss = 0.0
    perm = torch.randperm(n_train)
    optimizer.zero_grad()  # Reset gradients outside the sub-batch loop
    
    for i, start in enumerate(range(0, n_train, PRETRAIN_BATCH_SIZE)):
        idx = perm[start:start + PRETRAIN_BATCH_SIZE]
        xb = X_fit_bins_t[idx].long().to(DEVICE)   # upcast to int64 only for this batch
        yb = y_fit_tensor[idx].to(DEVICE)
        
        # Mixed Precision Forward Pass
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Scale the loss since we are accumulating gradients
            scaled_loss = loss / ACCUMULATION_STEPS
            
        # Backward Pass via scaler
        scaler.scale(scaled_loss).backward()
        
        # Update weights only after ACCUMULATION_STEPS iterations
        if (i + 1) % ACCUMULATION_STEPS == 0 or (start + PRETRAIN_BATCH_SIZE) >= n_train:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * len(yb)  # Track raw loss for reporting
        print(f"loss={total_loss}")

    scheduler.step()

    val_pred = predict_in_batches(model, X_val_bins_t)
    val_spearman = spearman(y_val_raw, val_pred)
    print(f"epoch {epoch + 1}/{PRETRAIN_EPOCHS}  train_loss={total_loss / n_train:.4f}  val_spearman={val_spearman:.4f}")

    if val_spearman > best_val_spearman:
        best_val_spearman = val_spearman
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
print(f"\nBest held-out Spearman during pretraining: {best_val_spearman:.4f}")


save_checkpoint(
    CHECKPOINT_PATH, model,
    config={**PRETRAIN_CONFIG, "n_features": len(feature_columns)},
    feature_columns=feature_columns,
    target_mean=target_mean, target_std=target_std,
)
print(f"Saved checkpoint to {CHECKPOINT_PATH}")
print("Upload this file as a resource alongside competition_transformer.ipynb.")