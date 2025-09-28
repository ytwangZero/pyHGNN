"""
Model & hyperparameter setup for an HGT-based HGNN on the MRN subgraph.
- Builds a configurable HGT encoder using torch & torch_geometric.
- Can derive metadata (node/edge types) and input dims from the NPZ
  produced by prepare_graph.py (graph_package.npz).

This file ONLY defines model + hyperparameters (no training loop).

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import inspect

try:
    from torch_geometric.nn import HGTConv
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for HGTConv. Install with: pip install torch-geometric"
    ) from e


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
            return HGTConv(**kwargs)

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
            # Post-layer transformations
            for nt in h:
                if self.norms is not None:
                    h[nt] = self.norms[layer_idx][nt](h[nt])
                h[nt] = self.act(h[nt])
                h[nt] = self.dropout(h[nt])

        # Output projection
        out = {nt: self.out_lin[nt](h[nt]) for nt in h}
        return out

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

