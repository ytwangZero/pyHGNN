"""
Model & hyperparameter setup for an HGT-based HGNN on the MRN subgraph.
- Builds a configurable HGT encoder using torch & torch_geometric.
- Can derive metadata (node/edge types) and input dims from the NPZ
  produced by prepare_graph.py (graph_package.npz).

This file ONLY defines model + hyperparameters (no training loop).

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import inspect
from collections import OrderedDict
import math
from torch_geometric.nn import HGTConv
from torch import Tensor
from torch_geometric.nn.conv.hgt_conv import HGTConv as _PyGHGTConv
from torch_geometric.utils.hetero import construct_bipartite_edge_index
from torch_geometric.utils import softmax

# --- NEW: an HGTConv that exposes per-edge, per-head attention (α) --------------------------------------------------------------------------

class HGTConvWithAtt(_PyGHGTConv):
    """
    Same API as torch_geometric.nn.HGTConv, but stores:
      - last_att: {edge_type(tuple): Tensor[E_rel, heads]}  (softmax-ed α)
      - last_edge_index: {edge_type: LongTensor[2, E_rel]}  (bipartite global idx)
      - last_offsets: (src_offset_dict, dst_offset_dict)    (for local id recovery)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_att: Dict[Tuple[str,str,str], Tensor] = {} # store attention score
        self.last_edge_index: Dict[Tuple[str,str,str], Tensor] = {}
        self.last_offsets: Optional[Tuple[Dict[str,int], Dict[str,int]]] = None

    def forward(self, x_dict, edge_index_dict):
        # --- 1) K,Q,V over node types (same as PyG) ---
        F, H = self.out_channels, self.heads
        D = F // H                                           # dimension per head
        kqv_dict = self.kqv_lin(x_dict)
        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        for key, val in kqv_dict.items():                    # extract K Q V for each node
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        # --- 2) 固定 edge_type 顺序，保证可复原每个关系的边切片 ---
        ordered_edge_types = [et for et in self.edge_types if et in edge_index_dict]
        ordered_eidict = OrderedDict((et, edge_index_dict[et]) for et in ordered_edge_types)

        # 目标/源的拼接偏移量（复原局部 id 用）
        def _cat(xd):
            cumsum = 0; outs = []; offset = {}
            for key, xx in xd.items():
                outs.append(xx); offset[key] = cumsum; cumsum += xx.size(0)
            return torch.cat(outs, dim=0), offset

        q, dst_offset = _cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict, ordered_eidict)

        # 记录偏移量，后面把全局 idx 还原为类型内局部 idx 会用到
        self.last_offsets = (src_offset, dst_offset)

        # 统一的二部图 edge_index
        edge_index, edge_attr = construct_bipartite_edge_index(
            ordered_eidict, src_offset, dst_offset, edge_attr_dict=self.p_rel, num_nodes=k.size(0)
        )

        # 根据 ordered_edge_types 逐一计算每种关系的边数，建立切片
        rel_slices = {}
        start = 0
        for et in ordered_edge_types:
            e = ordered_eidict[et]
            num_e = e.size(1) if isinstance(e, torch.Tensor) else e.nnz()
            rel_slices[et] = (start, start + num_e)
            start += num_e

        # --- 3) propagate（会调用message），并缓存每个关系的 α 与边 ---
        # 为了拿到 α，我们在 message 里把 softmax 后的 α 存在 self._alpha_tmp
        self._alpha_tmp = None  # will hold [E_total, heads]
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        # 拆回每个关系的 α 与 edge_index（按我们定义的顺序切片）
        self.last_att = {}
        self.last_edge_index = {}
        if self._alpha_tmp is not None:
            alpha_all = self._alpha_tmp  # [E_total, heads]
            for et in ordered_edge_types:
                s, e = rel_slices[et]
                self.last_att[et] = alpha_all[s:e].detach()
                self.last_edge_index[et] = edge_index[:, s:e].detach()

        # --- 4) 将 propagate 的输出重新组织为字典格式 ---
        # propagate 返回的是按目标类型拼接后的张量，这里用 dst_offset 切回每个节点类型
        out_dict = {}
        for node_type, offset in dst_offset.items():
            num_nodes = x_dict[node_type].size(0)
            out_dict[node_type] = out[offset:offset + num_nodes]

        # --- 5) 后处理与 skip（严格按 PyG HGTConv 的做法） ---
        # 先对每种类型做 gelu，然后一次性用 HeteroDictLinear 处理整个字典
        gelu_dict = {k: torch.nn.functional.gelu(v) if v is not None else v
                     for k, v in out_dict.items()}
        a_dict = self.out_lin(gelu_dict)  # HeteroDictLinear: 接收/返回 dict[str, Tensor]

        # skip-connection 并返回同构字典
        out_final = {}
        for node_type, out_ in a_dict.items():
            if out_.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out_ = alpha * out_ + (1 - alpha) * x_dict[node_type]
            out_final[node_type] = out_
        return out_final

    # 重写 message：把 softmax 后的 α 暂存下来
    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)       # [E_total, heads]
        self._alpha_tmp = alpha                          # 缓存整个批次的 α
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

# ------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------
# Hyperparameter configuration
# -----------------------------
@dataclass
class HGTConfig:
    # Architecture
    hidden_dim: int = 64          # shared hidden size per node type per layer
    out_dim: int = 64             # final embedding size (after last projection)
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    act: str = "gelu"              # "relu" | "gelu" | "elu"
    layer_norm: bool = True        # per-type LayerNorm after each layer
    add_self_loops: bool = False   # set True only if your edge_index_dict lacks self loops

    # Regularization helpers (used later in training loop; defined here for completeness)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    def make_activation(self) -> nn.Module:
        if self.act.lower() == "relu":
            return nn.ReLU()
        if self.act.lower() == "elu":
            return nn.ELU()
        return nn.GELU()


# -----------------------------
# HGT Encoder
# -----------------------------
class HGTEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder.

    Parameters
    ----------
    metadata : Tuple[List[str], List[Tuple[str, str, str]]]
        (node_types, edge_types) as required by torch_geometric's HGTConv.
    in_dims : Dict[str, int]
        Input feature dim per node type, e.g., {"feature": 2, "metabolite": 3}.
    cfg : HGTConfig
        Hyperparameters.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        cfg: HGTConfig = HGTConfig(),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types

        # Per-type input projection to hidden_dim
        self.in_lin = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], cfg.hidden_dim) for ntype in node_types
        })

        # HGT layers (version-adaptive arguments)
        def _make_hgtconv(metadata, cfg):
            params = inspect.signature(HGTConv.__init__).parameters
            kwargs = dict(
                in_channels=cfg.hidden_dim,
                out_channels=cfg.hidden_dim,
                metadata=metadata,
                heads=cfg.heads,
            )
            if 'dropout' in params:
                kwargs['dropout'] = cfg.dropout
            if 'group' in params:
                kwargs['group'] = 'sum'
            # return HGTConv(**kwargs)
# --------------------------------------------------------------------------------------------------------------------------------------
            return HGTConvWithAtt(**kwargs) 
# --------------------------------------------------------------------------------------------------------------------------------------

        # Create multiple HGT layers
        self.layers = nn.ModuleList([
            _make_hgtconv(metadata, cfg) for _ in range(cfg.num_layers)
        ])

        # Optional per-type LayerNorm after each layer
        if cfg.layer_norm:
            self.norms = nn.ModuleList([
                nn.ModuleDict({nt: nn.LayerNorm(cfg.hidden_dim) for nt in node_types})
                for _ in range(cfg.num_layers)
            ])
        else:
            self.norms = None

        self.dropout = nn.Dropout(cfg.dropout)
        self.act = cfg.make_activation()

        # Final per-type output projection (hidden_dim -> out_dim)
        self.out_lin = nn.ModuleDict({
            ntype: nn.Linear(cfg.hidden_dim, cfg.out_dim) for ntype in node_types # 线性变换
        })

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier init for linears
        for mod in list(self.in_lin.values()) + list(self.out_lin.values()):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        # HGT layers have their own initializers

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], edge_attr_dict: Dict[Tuple[str,str,str], torch.Tensor] | None = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_dict : dict
            Node-type keyed features, e.g. {"feature": [Nf, Ff], "metabolite": [Nm, Fm]}.
        edge_index_dict : dict
            Canonical edge dict keyed by (src_type, rel_type, dst_type) -> LongTensor[2, E].
        edge_attr_dict : dict | None
            Optional edge attributes per relation (not required for basic HGT).

        Returns
        -------
        out_dict : dict
            Per-type embeddings with shape [N_type, cfg.out_dim].
        """
        # Input projection
        h = {nt: self.in_lin[nt](x) for nt, x in x_dict.items()}

        att_collector: Dict[Tuple[str, str, str], torch.Tensor] = {}

        # HGT stack
        for layer_idx, conv in enumerate(self.layers):
            h = conv(h, edge_index_dict)
# -----------------------------------------------------------------------------------------------------------------------------------
            if hasattr(conv, "last_att") and conv.last_att:
                self.last_attention = {
                    "att": conv.last_att,
                    "edge_index": conv.last_edge_index,
                    "offsets": conv.last_offsets,
                }
# ------------------------------------------------------------------------------------------------------------------------------------

            # Post-layer transformations
            for nt in h:
                if self.norms is not None:
                    h[nt] = self.norms[layer_idx][nt](h[nt])
                h[nt] = self.act(h[nt])
                h[nt] = self.dropout(h[nt])
        

        # Output projection
        out = {nt: self.out_lin[nt](h[nt]) for nt in h}
        return out

# ------------------------------------------------------------------------------------------------------------------------------------  
    def get_last_attention(self):
            return getattr(self, "last_attention", None)
# ------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------
# Metadata & dims helpers
# -----------------------------
NPZ_EL_COUNT_KEY = "EL_count"
NPZ_EL_NAME_TMPL = "EL_{k}_name"
NPZ_X_FEATURE = "X_feature"
NPZ_X_MET = "X_metabolite"


def derive_metadata_from_npz(npz_path: str) -> Tuple[Tuple[List[str], List[Tuple[str, str, str]]], Dict[str, int]]:
    """Derive (metadata, in_dims) from graph_package.npz created by prepare_graph.py.

    Returns
    -------
    metadata : (node_types, edge_types)
        node_types: ["feature", "metabolite"],
        edge_types: list of (src, rel, dst) tuples reconstructed from EL names.
    in_dims : dict
        {"feature": Ff, "metabolite": Fm}
    """
    npz = np.load(npz_path, allow_pickle=True)

    # Node types are known from the package structure
    node_types = ["feature", "metabolite"]
    in_dims = {
        "feature": int(npz[NPZ_X_FEATURE].shape[1]),
        "metabolite": int(npz[NPZ_X_MET].shape[1]),
    }

    # Edge types from EL_*_name fields    (stored as "tgt|src|rel")
    num_el = int(npz[NPZ_EL_COUNT_KEY][0])
    edge_types_set = set()
    for k in range(num_el):
        name = str(npz[NPZ_EL_NAME_TMPL.format(k=k)][0])
        tgt, src, rel = name.split("|")
        edge_types_set.add((src, rel, tgt))  # PyG canonical order: (src, rel, dst)

    edge_types = sorted(edge_types_set)
    metadata = (node_types, edge_types)
    return metadata, in_dims


def build_model_from_npz(npz_path: str, cfg: HGTConfig | None = None) -> Tuple[HGTEncoder, Tuple[List[str], List[Tuple[str, str, str]]], Dict[str, int]]:
    """Convenience: derive metadata & dims, then instantiate HGTEncoder."""
    metadata, in_dims = derive_metadata_from_npz(npz_path)
    cfg = cfg or HGTConfig()
    model = HGTEncoder(metadata=metadata, in_dims=in_dims, cfg=cfg)
    return model, metadata, in_dims


# -----------------------------
# Device helper (optional)
# -----------------------------

def auto_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

