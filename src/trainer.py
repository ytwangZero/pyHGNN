# -*- coding: utf-8 -*-
"""
Full-graph training loop for the MRN HGNN

This module builds x_dict / edge_index_dict from NPZ, performs full-graph forward,
computes losses with negative sampling, tracks validation AUC/AP & L_edge, and
supports early stopping. It also exports embeddings and edge-level interaction
scores (as attention proxies) for the f→m relation.

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model_and_hparams import HGTConfig, HGTEncoder, build_model_from_npz, auto_device
from losses import LossConfig, compute_total_loss


# -----------------------------
# Training configuration
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 300
    patience: int = 20
    lr: float = 1e-3 # 控制每次参数更新的步长
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    monitor: str = "val_L_edge"      # or "val_AUC_fm" / "val_AP_fm" (case-insensitive)
    minimize: bool = True             # if monitor is loss -> True; if AUC/AP -> False
    val_ratio: float = 0.2
    seed: int = 42

    # Model hparams
    h_dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2

    # Loss hparams (see LossConfig for details)
    tau: float = 1.0
    lambda_edge: float = 1.0
    lambda_attr: float = 0.0
    lambda_hub: float = 0.05
    fm_neg_ratio: int = 1
    mm_neg_ratio: int = 1
    fm_importance_alpha: float = 0.5
    mm_anno_edge_boost: float = 0.3


# -----------------------------
# Helpers: NPZ -> tensors & dicts
# -----------------------------
NPZ_EL_COUNT_KEY = "EL_count"
NPZ_EL_NAME_TMPL = "EL_{k}_name"
NPZ_EL_PAIRS_TMPL = "EL_{k}_pairs"
NPZ_EL_W_TMPL = "EL_{k}_w"

# npz_path = "result"
def _rebuild_from_npz(npz_path: str, device: torch.device):
    """Return
    x_dict: {ntype: Tensor[N, F]}
    edge_index_full: {(src, rel, dst): LongTensor[2, E]}
    edge_weight_full: {(src, rel, dst): FloatTensor[E]}
    id_lists: {"feature": list[str], "metabolite": list[str]}
    rel_keys: dict with 'fm' and 'mm' canonical keys if present
    """
    npz = np.load(npz_path, allow_pickle=True)

    # Node features
    Xf = torch.tensor(npz["X_feature"], dtype=torch.float32, device=device)
    Xm = torch.tensor(npz["X_metabolite"], dtype=torch.float32, device=device)
    x_dict = {"feature": Xf, "metabolite": Xm}

    # Ids for mapping back
    ids_f = [str(x) for x in npz["ids_feature"].tolist()]
    ids_m = [str(x) for x in npz["ids_metabolite"].tolist()]
    id_lists = {"feature": ids_f, "metabolite": ids_m}

    # Relations
    nrel = int(npz[NPZ_EL_COUNT_KEY][0])
    edge_index_full: Dict[Tuple[str,str,str], torch.Tensor] = {}
    edge_weight_full: Dict[Tuple[str,str,str], torch.Tensor] = {}

    for k in range(nrel):
        name = str(npz[NPZ_EL_NAME_TMPL.format(k=k)][0])  # "tgt|src|rel"
        tgt, src, rel = name.split("|")
        pairs = torch.tensor(npz[NPZ_EL_PAIRS_TMPL.format(k=k)], dtype=torch.long, device=device)
        # Convert stored [tgt, src] to canonical (src, rel, dst)
        ei = torch.stack([pairs[:,1], pairs[:,0]], dim=0)  # [2, E]
        w = torch.tensor(npz[NPZ_EL_W_TMPL.format(k=k)], dtype=torch.float32, device=device)
        edge_index_full[(src, rel, tgt)] = ei
        edge_weight_full[(src, rel, tgt)] = w

    # Pick canonical relations
    rel_fm = None
    rel_mm = None
    for key in edge_index_full.keys():
        src, rel, dst = key
        if src == 'feature' and dst == 'metabolite' and rel.startswith('f_'):
            rel_fm = key
        if src == 'metabolite' and dst == 'metabolite' and ('m_m' in rel):
            if rel == 'm_m':
                rel_mm = key
    if rel_mm is None:
        for key in edge_index_full.keys():
            if key[0]=='metabolite' and key[2]=='metabolite':
                rel_mm = key; break

    rel_keys = {"fm": rel_fm, "mm": rel_mm}
    return x_dict, edge_index_full, edge_weight_full, id_lists, rel_keys


def _split_train_val(ei: torch.Tensor, ew: torch.Tensor, val_ratio: float, seed: int):
    E = ei.size(1)
    g = torch.Generator(device=ei.device)
    g.manual_seed(seed)
    perm = torch.randperm(E, generator=g)
    n_val = max(1, int(E * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    def sub(ei, idx):
        return ei.index_select(1, idx), ew.index_select(0, idx)
    return sub(ei, train_idx), sub(ei, val_idx)


# -----------------------------
# Metrics: ROC-AUC & Average Precision from pos/neg scores
# -----------------------------
@torch.no_grad() # 禁用梯度计算，节省内存和加速计算
def _auc_ap_from_scores(pos: torch.Tensor, neg: torch.Tensor) -> Tuple[float, float]:
    # Concatenate
    y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    s = torch.cat([pos, neg])
    # AUC via Mann-Whitney U statistic
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, order.numel()+1, device=order.device, dtype=torch.float64)
    n_pos = y.sum().item()
    n_neg = y.numel() - n_pos
    sum_ranks_pos = ranks[y.bool()].sum().item()
    auc = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg + 1e-8)
    # AP (area under precision-recall curve)
    order_desc = torch.argsort(s, descending=True)
    y_sorted = y[order_desc]
    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1 - y_sorted, dim=0)
    precision = tp / (tp + fp + 1e-8)
    ap = (precision[y_sorted.bool()].sum() / (n_pos + 1e-8)).item()
    return float(auc), float(ap)


@torch.no_grad()
def _predict_edge_scores(h_src: torch.Tensor, h_dst: torch.Tensor, ei: torch.Tensor, tau: float) -> torch.Tensor:
    i, j = ei[0], ei[1]
    logits = (h_src.index_select(0, i) * h_dst.index_select(0, j)).sum(dim=-1) / max(tau, 1e-8)
    return torch.sigmoid(logits)


# -----------------------------
# Early stopping helper
# -----------------------------
class EarlyStop:
    def __init__(self, patience: int, minimize: bool) -> None:
        self.patience = patience
        self.minimize = minimize
        self.best_score = math.inf if minimize else -math.inf
        self.best_state = None
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        improved = (value < self.best_score) if self.minimize else (value > self.best_score)
        if improved:
            self.best_score = value
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.patience


# -----------------------------
# Core training routine
# -----------------------------
@torch.no_grad()
def _val_metrics(out_dict, edge_index_dict_train, edge_index_dict_val, rel_keys, tau):
    h_f, h_m = out_dict['feature'], out_dict['metabolite']
    metrics = {}

    # f→m
    if rel_keys['fm'] is not None and rel_keys['fm'] in edge_index_dict_val:
        ei_pos = edge_index_dict_val[rel_keys['fm']]
        # sample same number of negatives uniformly
        E = ei_pos.size(1)
        neg_i = torch.randint(0, h_f.size(0), (E,), device=ei_pos.device)
        neg_j = torch.randint(0, h_m.size(0), (E,), device=ei_pos.device)
        pos = _predict_edge_scores(h_f, h_m, ei_pos, tau)
        neg = _predict_edge_scores(h_f, h_m, torch.stack([neg_i, neg_j], dim=0), tau)
        auc, ap = _auc_ap_from_scores(pos, neg)
        metrics['AUC_fm'] = auc
        metrics['AP_fm']  = ap

    # m–m
    if rel_keys['mm'] is not None and rel_keys['mm'] in edge_index_dict_val:
        ei_pos = edge_index_dict_val[rel_keys['mm']]
        E = ei_pos.size(1)
        neg_i = torch.randint(0, h_m.size(0), (E,), device=ei_pos.device)
        neg_j = torch.randint(0, h_m.size(0), (E,), device=ei_pos.device)
        pos = _predict_edge_scores(h_m, h_m, ei_pos, tau)
        neg = _predict_edge_scores(h_m, h_m, torch.stack([neg_i, neg_j], dim=0), tau)
        auc, ap = _auc_ap_from_scores(pos, neg)
        metrics['AUC_mm'] = auc
        metrics['AP_mm']  = ap

    return metrics


def run_training(npz_path: str, out_dir: str, cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = auto_device()

    # 1) Data
    x_dict, edge_index_full, edge_weight_full, id_lists, rel_keys = _rebuild_from_npz(npz_path, device)

    # Build loss edge dicts (train/val) for only canonical relations
    edge_index_train = {}
    edge_index_val   = {}
    edge_weight_train = {}
    edge_weight_val   = {}

    for key_name in ['fm', 'mm']:
        key = rel_keys[key_name]
        if key is None:
            continue
        (ei_tr, ew_tr), (ei_va, ew_va) = _split_train_val(edge_index_full[key], edge_weight_full[key], cfg.val_ratio, cfg.seed)
        edge_index_train[key] = ei_tr
        edge_weight_train[key] = ew_tr
        edge_index_val[key] = ei_va
        edge_weight_val[key] = ew_va

    # 2) Model
    mcfg = HGTConfig(hidden_dim=cfg.h_dim, out_dim=cfg.h_dim, num_layers=cfg.layers,
                     heads=cfg.heads, dropout=cfg.dropout)
    model, metadata, in_dims = build_model_from_npz(npz_path, mcfg)
    model = model.to(device)

    # 3) Optimizer
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 4) Loss config
    lcfg = LossConfig(
        tau=cfg.tau,
        lambda_edge=cfg.lambda_edge,
        lambda_attr=cfg.lambda_attr,
        lambda_hub=cfg.lambda_hub,
        fm_neg_ratio=cfg.fm_neg_ratio,
        mm_neg_ratio=cfg.mm_neg_ratio,
        fm_importance_alpha=cfg.fm_importance_alpha,
        mm_anno_edge_boost=cfg.mm_anno_edge_boost,
    )

    # 5) Early stopping
    monitor = cfg.monitor.lower()
    minimize = cfg.minimize
    stopper = EarlyStop(patience=cfg.patience, minimize=minimize)

    best_logs = {}

    # 6) Training loop (full-graph forward)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x_dict, edge_index_full)
        # train loss uses only train relations
        loss_dict = compute_total_loss(
            out_dict=out,
            edge_index_dict=edge_index_train,
            edge_weight_dict=edge_weight_train,
            X_feature=x_dict['feature'],
            X_metabolite=x_dict['metabolite'],
            loss_cfg=lcfg,
        )
        loss = loss_dict['total']
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out_val = model(x_dict, edge_index_full)  # same graph forward
            # compute validation edge loss on val splits
            val_loss_dict = compute_total_loss(
                out_dict=out_val,
                edge_index_dict=edge_index_val,
                edge_weight_dict=edge_weight_val,
                X_feature=x_dict['feature'],
                X_metabolite=x_dict['metabolite'],
                loss_cfg=lcfg,
            )
            metrics = _val_metrics(out_val, edge_index_train, edge_index_val, rel_keys, tau=cfg.tau)

        logs = {
            'epoch': epoch,
            'train_total': float(loss_dict['total'].detach().cpu()),
            'val_total': float(val_loss_dict['total'].detach().cpu()),
            'val_L_edge': float(val_loss_dict['L_edge'].detach().cpu()),
            **{k: float(v) for k, v in metrics.items()},
        }

        # Select monitored value
        monitor_value = logs.get(cfg.monitor, logs['val_L_edge'])
        stop = stopper.step(monitor_value, model, epoch)

        # Print short log
        msg = (f"[E{epoch:03d}] train={logs['train_total']:.4f} "
               f"val={logs['val_total']:.4f} L_edge={logs['val_L_edge']:.4f} "
               f"AUC_fm={logs.get('AUC_fm', float('nan')):.3f} "
               f"AP_fm={logs.get('AP_fm', float('nan')):.3f}"
               f"AUC_mm={logs.get('AUC_mm', float('nan')):.3f} "
               f"AP_mm={logs.get('AP_mm', float('nan')):.3f}"
               )
        print(msg)

        if stop:
            best_logs = logs
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch} (monitor={cfg.monitor}, best={stopper.best_score:.4f})")
            break
        else:
            best_logs = logs

    # Load best state
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    # Final forward for outputs
    model.eval()
    with torch.no_grad():
        out_final = model(x_dict, edge_index_full)

    # Edge scores (attention proxy) for f→m if present
    fm_scores_df = None
    if rel_keys['fm'] is not None:
        ei = edge_index_full[rel_keys['fm']].detach().cpu()
        scores = _predict_edge_scores(out_final['feature'], out_final['metabolite'], edge_index_full[rel_keys['fm']], tau=cfg.tau).detach().cpu().numpy()
        fi = ei[0].numpy(); mj = ei[1].numpy()
        rows = []
        ids_f = id_lists['feature']; ids_m = id_lists['metabolite']
        for s, i, j in zip(scores, fi, mj):
            rows.append((ids_f[int(i)], ids_m[int(j)], float(s)))
        import pandas as pd
        fm_scores_df = pd.DataFrame(rows, columns=["feature_id", "metabolite_id", "edge_score"])


    # Edge scores (attention proxy) for m–m if present
    mm_scores_df = None
    if rel_keys['mm'] is not None:
        ei = edge_index_full[rel_keys['mm']].detach().cpu()
        scores = _predict_edge_scores(
            out_final['metabolite'], out_final['metabolite'],
            edge_index_full[rel_keys['mm']], tau=cfg.tau
        ).detach().cpu().numpy()
        mi = ei[0].numpy(); mj = ei[1].numpy()
        rows = []
        ids_m = id_lists['metabolite']
        for s, i, j in zip(scores, mi, mj):
            rows.append((ids_m[int(i)], ids_m[int(j)], float(s)))
        import pandas as pd
        mm_scores_df = pd.DataFrame(rows, columns=["metabolite_id_src", "metabolite_id_dst", "edge_score"])

    # Embeddings
    emb_dict = {k: v.detach().cpu().numpy() for k, v in out_final.items()}


    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_f = pd.DataFrame(emb_dict['feature'])
    df_f.insert(0, "feature_id", id_lists["feature"])
    df_f.to_csv(out_dir / "emb_feature.csv", index=False)

    df_m = pd.DataFrame(emb_dict['metabolite'])
    df_m.insert(0, "metabolite_id", id_lists["metabolite"])
    df_m.to_csv(out_dir / "emb_metabolite.csv", index=False)

    if fm_scores_df is not None:
        fm_scores_df.to_csv(out_dir / "fm_edge_scores.csv", index=False)
    if mm_scores_df is not None:
        mm_scores_df.to_csv(out_dir / "mm_edge_scores.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------------------------
    print("DEBUG att:", model.get_last_attention() is not None)

    # === NEW: 导出真正的 HGT 注意力 (α) ===
    att_pack = model.get_last_attention()
    if att_pack is not None:
        att_dict = att_pack['att']
        eidx_dict = att_pack['edge_index']
        offsets = att_pack['offsets']

    def _export_att(rel_key, ids_src, ids_dst, fname):
        if rel_key is None or rel_key not in att_dict:
            return None
        alpha = att_dict[rel_key].detach().cpu()          # [E, H]

        # 用原始传入的 canonical edge_index（local 下标，和 ids_* 一致）
        ei_canon = edge_index_full[rel_key].detach().cpu()  # [2, E]
        # 保险检查：两者边数一致
        if alpha.size(0) != ei_canon.size(1):
            print(f"[WARN] attention rows ({alpha.size(0)}) != edges ({ei_canon.size(1)}) for {rel_key}")
        src_local = ei_canon[0].numpy()
        dst_local = ei_canon[1].numpy()

        # 多头聚合（可改成保存每个 head）
        w = alpha.mean(dim=1).numpy()

        rows = [(ids_src[int(i)], ids_dst[int(j)], float(s)) for s, i, j in zip(w, src_local, dst_local)]
        import pandas as pd
        df = pd.DataFrame(rows, columns=[f"{rel_key[0]}_id", f"{rel_key[2]}_id", "att_weight"])
        df.to_csv(out_dir / fname, index=False)
        return df


    # feature → metabolite
    if rel_keys['fm'] is not None:
        _export_att(rel_keys['fm'], id_lists['feature'], id_lists['metabolite'], "fm_attention.csv")

    # metabolite ↔ metabolite
    if rel_keys['mm'] is not None:
        _export_att(rel_keys['mm'], id_lists['metabolite'], id_lists['metabolite'], "mm_attention.csv")
# ------------------------------------------------------------------------------------------------------------------------------------


    return {
        'best_epoch': stopper.best_epoch if stopper.best_epoch != -1 else cfg.epochs,
        'best_monitor': stopper.best_score,
        'emb_dict': emb_dict,
        'fm_edge_scores': fm_scores_df,
        'mm_edge_scores': mm_scores_df,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--npz', type=str, default='result/graph_package.npz')
    p.add_argument('--out', type=str, default='result/outputs')
    p.add_argument('--monitor', type=str, default='val_L_edge')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=20)
    args = p.parse_args()

    cfg = TrainConfig(epochs=args.epochs, patience=args.patience, monitor=args.monitor,
                      minimize=(args.monitor.lower() in ["val_l_edge", "val_total", "val_loss"]))

    res = run_training(args.npz, args.out, cfg)
    print("Best epoch:", res['best_epoch'])
    print("Best monitor:", res['best_monitor'])
    if res['fm_edge_scores'] is not None:
        print("Saved fm_edge_scores.csv with", len(res['fm_edge_scores']), "rows")
    if res['mm_edge_scores'] is not None:
        print("Saved mm_edge_scores.csv with", len(res['mm_edge_scores']), "rows")


# python src/trainer.py --npz result/graph_package.npz --out result/outputs --monitor val_L_edge --epochs 300 --patience 20
