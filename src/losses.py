# -*- coding: utf-8 -*-
"""
Losses for HGNN on MRN subgraph (Task D):
- Edge reconstruction with per-relation weighting (rewards important feature→metabolite edges)
- Optional attribute alignment (weak)
- Hub penalty (discourage overly strong hubs)
- Annotated cohesion (encourage annotated metabolites to cluster)

This module is model-agnostic: it only consumes embeddings produced by the
HGT encoder and the hetero edge_index_dict. It is designed to work with the
NPZ produced by prepare_graph.py and the model from model_and_hparams.py.

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Loss configuration
# -----------------------------
@dataclass
class LossConfig:
    # Temperatures
    tau: float = 1.0                 # scale for dot-product logits

    # Weights of loss components
    lambda_edge: float = 1.0 # 边预测是主要任务
    lambda_attr: float = 0.05 # 暂时不使用属性对齐
    lambda_hub: float = 0.05 # 轻微的中心节点正则化  

    # Negative sampling ratios (per positive)
    fm_neg_ratio: int = 1
    mm_neg_ratio: int = 1

    # Edge reweighting (reward important features; boost anno-anno m–m)
    fm_importance_alpha: float = 0.5   # weight multiplier: (1 + alpha*imp_f)
    mm_anno_edge_boost: float = 0.3    # extra weight if both endpoints annotated

    # Attribute alignment (weak; optional)
    attr_mode: str = "norm"            # how to form target attribute signal from features


# -----------------------------
# Utilities
# -----------------------------
BCEWL = nn.BCEWithLogitsLoss(reduction='none')


def _pair_dot(h_src: torch.Tensor, idx_src: torch.Tensor, h_dst: torch.Tensor, idx_dst: torch.Tensor, tau: float) -> torch.Tensor:
    """Compute per-edge logits = (h_src[i]·h_dst[j]) / tau for pairs defined by indices."""
    z = (h_src.index_select(0, idx_src) * h_dst.index_select(0, idx_dst)).sum(dim=-1)
    return z / max(tau, 1e-8)


def _sample_neg_edges(num_pos: int, num_src: int, num_dst: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample negative edges as random pairs (allowing collisions)."""
    i = torch.randint(0, num_src, (num_pos,), device=device)
    j = torch.randint(0, num_dst, (num_pos,), device=device)
    return i, j


def _node_degrees(edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Compute (approx) degrees per node type by summing degrees over all relations.
    Returns degree tensors per node type, shape [N_type]."""
    deg = {nt: torch.zeros(sizes[nt], dtype=torch.float32, device=_any_device(edge_index_dict)) for nt in sizes}
    for (src, rel, dst), ei in edge_index_dict.items():
        src_idx, dst_idx = ei[0], ei[1]
        deg[src].index_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float32))
        deg[dst].index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float32))
    return deg


def _any_device(edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.device:
    for v in edge_index_dict.values():
        return v.device
    return torch.device('cpu')


def _feature_importance(X_feature: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Combine feature attributes into a scalar importance in [0,1].
    Expect X_feature[:, 0] = weight_p, X_feature[:, 1] = weight_fc.
    """
    if X_feature is None:
        return None
    x = X_feature
    if x.dim() != 2 or x.size(1) < 2:
        return None
    imp = torch.sqrt(x[:, 0] * x[:, 1])
    return torch.clamp(imp, 0.0, 1.0)


def _is_annotated(X_met: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return boolean mask for annotated metabolites from X_metabolite[:, 2] == is_anno."""
    if X_met is None:
        return None
    if X_met.dim() != 2 or X_met.size(1) < 3:
        return None
    return (X_met[:, 2] > 0.5)


# -----------------------------
# Edge reconstruction losses
# -----------------------------

def edge_reconstruction_losses(
    out_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    X_feature: Optional[torch.Tensor] = None,
    X_metabolite: Optional[torch.Tensor] = None,
    loss_cfg: LossConfig = LossConfig(),
) -> Dict[str, torch.Tensor]:
    """Compute weighted edge reconstruction losses for f→m and m–m relations with negative sampling.

    Returns a dict with keys: 'L_fm', 'L_mm', 'L_edge'.
    """
    device = _any_device(edge_index_dict)
    h_f = out_dict.get('feature')
    h_m = out_dict.get('metabolite')
    assert h_f is not None and h_m is not None, "out_dict must contain 'feature' and 'metabolite'"

    # Pre-compute scalars
    imp_f = _feature_importance(X_feature)
    if imp_f is not None:
        imp_f = imp_f.to(device)
    anno_mask = _is_annotated(X_metabolite)
    if anno_mask is not None:
        anno_mask = anno_mask.to(device)

    # Find relations if present
    rel_fm = None
    rel_mm = None
    for k in edge_index_dict.keys():
        src, rel, dst = k
        if src == 'feature' and dst == 'metabolite' and rel.startswith('f_'):
            rel_fm = k
        if src == 'metabolite' and dst == 'metabolite' and ('m_m' in rel):
            # pick the forward 'm_m' if available; otherwise any m–m
            if rel == 'm_m':
                rel_mm = k
    # if we didn't find exact 'm_m', fallback to first meta-meta
    if rel_mm is None:
        for k in edge_index_dict.keys():
            if k[0] == 'metabolite' and k[2] == 'metabolite':
                rel_mm = k; break

    # Compute per-relation losses
    L_fm = torch.tensor(0.0, device=device)
    L_mm = torch.tensor(0.0, device=device)

    # 1) feature→metabolite
    if rel_fm is not None:
        ei = edge_index_dict[rel_fm]  # [2, E]
        pos_src, pos_dst = ei[0], ei[1]
        E = pos_src.numel()
        pos_w = None
        if edge_weight_dict is not None and rel_fm in edge_weight_dict:
            pos_w = edge_weight_dict[rel_fm].to(device)
        else:
            pos_w = torch.ones(E, device=device)

        # Reward important features by scaling positive weight
        if imp_f is not None and loss_cfg.fm_importance_alpha != 0.0:
            pos_w = pos_w * (1.0 + loss_cfg.fm_importance_alpha * imp_f.index_select(0, pos_src))

        # Positive logits
        pos_logits = _pair_dot(h_src=h_f, idx_src=pos_src, h_dst=h_m, idx_dst=pos_dst, tau=loss_cfg.tau)
        pos_labels = torch.ones_like(pos_logits)
        pos_loss = (BCEWL(pos_logits, pos_labels) * pos_w).mean()

        # Negative sampling
        kneg = max(1, int(loss_cfg.fm_neg_ratio))
        neg_losses = []
        for _ in range(kneg):
            neg_src, neg_dst = _sample_neg_edges(E, h_f.size(0), h_m.size(0), device)
            neg_logits = _pair_dot(h_src=h_f, idx_src=neg_src, h_dst=h_m, idx_dst=neg_dst, tau=loss_cfg.tau)
            neg_labels = torch.zeros_like(neg_logits)
            neg_w = torch.ones_like(neg_logits)
            neg_losses.append((BCEWL(neg_logits, neg_labels) * neg_w).mean())
        L_fm = pos_loss + (sum(neg_losses) / len(neg_losses)) if neg_losses else pos_loss

    # 2) metabolite–metabolite
    if rel_mm is not None:
        ei = edge_index_dict[rel_mm]
        pos_src, pos_dst = ei[0], ei[1]
        E = pos_src.numel()
        pos_w = None
        if edge_weight_dict is not None and rel_mm in edge_weight_dict:
            pos_w = edge_weight_dict[rel_mm].to(device)
        else:
            pos_w = torch.ones(E, device=device)

        # Boost anno–anno edges
        if anno_mask is not None and loss_cfg.mm_anno_edge_boost != 0.0:
            boost = (anno_mask.index_select(0, pos_src) & anno_mask.index_select(0, pos_dst)).float()
            pos_w = pos_w * (1.0 + loss_cfg.mm_anno_edge_boost * boost)

        pos_logits = _pair_dot(h_src=h_m, idx_src=pos_src, h_dst=h_m, idx_dst=pos_dst, tau=loss_cfg.tau)
        pos_labels = torch.ones_like(pos_logits)
        pos_loss = (BCEWL(pos_logits, pos_labels) * pos_w).mean()

        kneg = max(1, int(loss_cfg.mm_neg_ratio))
        neg_losses = []
        for _ in range(kneg):
            neg_src, neg_dst = _sample_neg_edges(E, h_m.size(0), h_m.size(0), device)
            neg_logits = _pair_dot(h_src=h_m, idx_src=neg_src, h_dst=h_m, idx_dst=neg_dst, tau=loss_cfg.tau)
            neg_labels = torch.zeros_like(neg_logits)
            neg_w = torch.ones_like(neg_logits)
            neg_losses.append((BCEWL(neg_logits, neg_labels) * neg_w).mean())
        L_mm = pos_loss + (sum(neg_losses) / len(neg_losses)) if neg_losses else pos_loss

    L_edge = L_fm + L_mm
    return {"L_fm": L_fm, "L_mm": L_mm, "L_edge": L_edge}


# -----------------------------
# Attribute alignment (weak/optional)
# -----------------------------

def attribute_alignment_loss(
    out_dict: Dict[str, torch.Tensor],
    X_feature: Optional[torch.Tensor] = None,
    X_metabolite: Optional[torch.Tensor] = None,
    mode: str = "norm",
) -> torch.Tensor:
    """Weak alignment between embedding norms and scalar attributes.
    - For features: target = mean(weight_p, weight_fc)
    - For metabolites: target = weight_b (bridge residual strength)
    """
    device = next(iter(out_dict.values())).device
    losses = []

    if X_feature is not None and 'feature' in out_dict:
        h = out_dict['feature']
        t = _feature_importance(X_feature.to(device))
        if t is not None:
            z = h.norm(p=2, dim=-1)
            z = (z - z.mean()) / (z.std() + 1e-6)
            t = (t - t.mean()) / (t.std() + 1e-6)
            losses.append(F.mse_loss(z, t))

    if X_metabolite is not None and 'metabolite' in out_dict:
        h = out_dict['metabolite']
        # use weight_b as cohesion proxy if available (col=1)
        if X_metabolite.size(1) >= 2:
            t = X_metabolite[:, 1].to(device)
            z = h.norm(p=2, dim=-1)
            z = (z - z.mean()) / (z.std() + 1e-6)
            t = (t - t.mean()) / (t.std() + 1e-6)
            losses.append(F.mse_loss(z, t))

    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


# -----------------------------
# Hub penalty
# -----------------------------

def hub_penalty(
    out_dict, 
    edge_index_dict,
    per_type_norm: bool = True
):
    device = next(iter(out_dict.values())).device
    sizes = {nt: out_dict[nt].size(0) for nt in out_dict}
    
    # 初始化度数为0
    deg = {nt: torch.zeros(sizes[nt], dtype=torch.float32, device=device) for nt in sizes}
    
    # 为不同节点类型选择不同的边类型计算度数
    for (src, rel, dst), ei in edge_index_dict.items():
        src_idx, dst_idx = ei[0], ei[1]
        
        # feature节点：只从f→m边计算度数
        if src == "feature" and dst == "metabolite":
            # feature节点作为源节点的度数
            deg["feature"].index_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float32))
            
        # metabolite节点：只从m→m边计算度数  
        elif src == "metabolite" and dst == "metabolite":
            # metabolite节点作为源节点的度数
            deg["metabolite"].index_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float32))
            # metabolite节点作为目标节点的度数
            deg["metabolite"].index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float32))
    
    losses = []
    for nt, h in out_dict.items():
        d = deg[nt]
        
        # 按类型归一化
        if per_type_norm:
            d = d / (d.max() + 1e-6)
        else:
            max_all = torch.max(torch.cat([deg[t] for t in deg]))
            d = d / (max_all + 1e-6)
        
        # 节点类型特异性惩罚系数
        if nt == "metabolite":
            coef = 1.0
        elif nt == "feature":
            coef = 0.3
        else:
            coef = 1.0
            
        losses.append(coef * (d * (h.pow(2).sum(dim=-1))).mean())
    
    return sum(losses) / len(losses)


# -----------------------------
# Aggregate total loss
# -----------------------------

def compute_total_loss(
    out_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    X_feature: Optional[torch.Tensor] = None,
    X_metabolite: Optional[torch.Tensor] = None,
    loss_cfg: LossConfig = LossConfig(),
) -> Dict[str, torch.Tensor]:
    comp = edge_reconstruction_losses(
        out_dict=out_dict,
        edge_index_dict=edge_index_dict,
        edge_weight_dict=edge_weight_dict,
        X_feature=X_feature,
        X_metabolite=X_metabolite,
        loss_cfg=loss_cfg,
    )

    L_attr = attribute_alignment_loss(
        out_dict=out_dict,
        X_feature=X_feature,
        X_metabolite=X_metabolite,
        mode=loss_cfg.attr_mode,
    )

    L_hub = hub_penalty(out_dict=out_dict, edge_index_dict=edge_index_dict)
    # L_anno = annotated_cohesion(out_dict=out_dict, X_metabolite=X_metabolite)

    total = (
        loss_cfg.lambda_edge * comp['L_edge'] +
        loss_cfg.lambda_attr * L_attr +
        loss_cfg.lambda_hub  * L_hub
    )

    return {
        'total': total,
        **comp,
        'L_attr': L_attr,
        'L_hub': L_hub
    }
