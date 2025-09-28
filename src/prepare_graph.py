# -*- coding: utf-8 -*-
"""
Prepare MRN heterogeneous subgraph for HGT training.
Reads CSVs exported from R and builds (feature, times, edge_list, indxs, edge_weight).
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ----------------------------
# Config: input CSV file paths
# ----------------------------
DATA_DIR = "data"
OUT_DIR = "result"

PATH_NODE_F    = os.path.join(DATA_DIR, "Node_F.csv")         # cols: ID, weight_p, weight_fc, type=feature
PATH_NODE_M    = os.path.join(DATA_DIR, "Node_M.csv")         # cols: ID, weight_d, weight_b, is_anno, type
PATH_EDGE_FM   = os.path.join(DATA_DIR, "Edge_FM.csv")        # cols: from (feature ID), to (metabolite INCHI), weight, type
PATH_EDGE_MM   = os.path.join(DATA_DIR, "Edge_MM.csv")        # cols: from INCHI, to INCHI, weight(=1), type

# ----------------------------
# Helpers
# ----------------------------
def _ensure_01(x: np.ndarray) -> np.ndarray:
    """Clamp to [0,1] just in case; NaNs -> 0."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)

def _make_edge_dict():
    return defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(list)  # relation_type -> list of [target_id, source_id]
        )
    )

def _make_weight_dict():
    return defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)  # relation_type -> list of weights (float)
        )
    )

# ----------------------------
# Load nodes
# ----------------------------
def load_nodes():
    # feature nodes
    nf = pd.read_csv(PATH_NODE_F, dtype={"ID": str})
    assert {"ID","weight_p","weight_fc"}.issubset(nf.columns), \
        "Node_F.csv must contain: ID, weight_p, weight_fc"

    # metabolite nodes 
    nm = pd.read_csv(PATH_NODE_M, dtype={"ID": str})
    assert {"ID","weight_d","weight_b","is_anno"}.issubset(nm.columns), \
        "Node_M must contain: ID, weight_d, weight_b, is_anno"

    # Drop duplicates, keep first
    nf = nf.drop_duplicates(subset=["ID"]).reset_index(drop=True)
    nm = nm.drop_duplicates(subset=["ID"]).reset_index(drop=True)

    # Build id->index maps
    feat_ids = nf["ID"].tolist()
    met_ids  = nm["ID"].tolist()
    feat_id2idx = {k:i for i,k in enumerate(feat_ids)}
    met_id2idx  = {k:i for i,k in enumerate(met_ids)}

    # Feature matrices
    X_feat = nf[["weight_p","weight_fc"]].to_numpy()
    X_met  = nm[["weight_d","weight_b","is_anno"]].to_numpy()

    X_feat = _ensure_01(X_feat)
    X_met  = _ensure_01(X_met)

    feature = {
        "feature": X_feat,          # shape [n_feat, 2]
        "metabolite": X_met         # shape [n_met, 3]
    }

    # times 占位（若无时间信息，用 1）
    times = {
        "feature": np.ones(X_feat.shape[0], dtype=np.float32),
        "metabolite": np.ones(X_met.shape[0], dtype=np.float32),
    }

    indxs = {
        "feature": np.arange(X_feat.shape[0], dtype=np.int64),
        "metabolite": np.arange(X_met.shape[0], dtype=np.int64),
        "feature_id2idx": feat_id2idx,
        "metabolite_id2idx": met_id2idx,
        "feature_ids": feat_ids,
        "metabolite_ids": met_ids,
    }

    return feature, times, indxs

# ----------------------------
# Load edges -> edge_list / edge_weight
# ----------------------------
def load_edges(indxs):
    edge_list   = _make_edge_dict()
    edge_weight = _make_weight_dict()

    feat_id2idx = indxs["feature_id2idx"]
    met_id2idx  = indxs["metabolite_id2idx"]

    # feature–metabolite
    efm = pd.read_csv(PATH_EDGE_FM, dtype={"from": str, "to": str})
    assert {"from","to","weight"}.issubset(efm.columns), "Edge_FM.csv must contain: from, to, weight"
    efm = efm[efm["from"].isin(feat_id2idx) & efm["to"].isin(met_id2idx)].copy()

    for _, row in efm.iterrows(): # _表示忽略索引；逐行遍历DataFrame
        f_idx = feat_id2idx[row["from"]]
        m_idx = met_id2idx[row["to"]]
        w = float(row["weight"]) if pd.notna(row["weight"]) else 0.0

        # 关系方向：target, source
        # 1) metabolite <- feature （f_m）
        edge_list["metabolite"]["feature"]["f_m"].append([m_idx, f_idx])
        edge_weight["metabolite"]["feature"]["f_m"].append(w)
        # 2) feature <- metabolite （rev_f_m）
        edge_list["feature"]["metabolite"]["rev_f_m"].append([f_idx, m_idx])
        edge_weight["feature"]["metabolite"]["rev_f_m"].append(w)

    # metabolite–metabolite
    emm = pd.read_csv(PATH_EDGE_MM, dtype={"from": str, "to": str})
    assert {"from","to","weight"}.issubset(emm.columns), "Edge_MM.csv must contain: from, to, weight"
    emm = emm[emm["from"].isin(met_id2idx) & emm["to"].isin(met_id2idx)].copy()

    # 若是无向边，这里添加双向；若已是有向，按需要调整
    for _, row in emm.iterrows(): 
        u = met_id2idx[row["from"]]
        v = met_id2idx[row["to"]]
        w = float(row["weight"]) if pd.notna(row["weight"]) else 1.0

        # m_m：v <- u
        edge_list["metabolite"]["metabolite"]["m_m"].append([v, u])
        edge_weight["metabolite"]["metabolite"]["m_m"].append(w)
        # rev_m_m：u <- v
        edge_list["metabolite"]["metabolite"]["rev_m_m"].append([u, v])
        edge_weight["metabolite"]["metabolite"]["rev_m_m"].append(w)

    return edge_list, edge_weight

# ----------------------------
# Entry
# ----------------------------
def build_graph_package(save_npz=True, out_path=os.path.join(OUT_DIR, "graph_package.npz")):
    feature, times, indxs = load_nodes()
    edge_list, edge_weight = load_edges(indxs)

    print("=== SUMMARY ===")
    print(f"#features:    {feature['feature'].shape[0]}  dim={feature['feature'].shape[1]}")
    print(f"#metabolites: {feature['metabolite'].shape[0]}  dim={feature['metabolite'].shape[1]}")
    # 统计边数
    def count_edges(ed):
        c = 0
        for tgt in ed:
            for src in ed[tgt]:
                for rel in ed[tgt][src]:
                    c += len(ed[tgt][src][rel])
        return c
    print(f"#edges total: {count_edges(edge_list)}")
    print("Relations present:")
    for tgt in edge_list:
        for src in edge_list[tgt]:
            for rel in edge_list[tgt][src]:
                print(f" - {tgt} <- {src} : {rel} ({len(edge_list[tgt][src][rel])})")

    if save_npz:
        # 为了跨脚本可读，这里将嵌套字典展开为扁平 npz；训练时再组装
        flat = {}
        # node features & times
        flat["X_feature"]     = feature["feature"]
        flat["X_metabolite"]  = feature["metabolite"]
        flat["T_feature"]     = times["feature"]
        flat["T_metabolite"]  = times["metabolite"]
        # indices & id lists（以便回写/对齐）
        flat["idx_feature"]   = indxs["feature"]
        flat["idx_metabolite"]= indxs["metabolite"]
        flat["ids_feature"]   = np.array(indxs["feature_ids"], dtype=object)
        flat["ids_metabolite"]= np.array(indxs["metabolite_ids"], dtype=object)

        # 将 edge_list / edge_weight 展平为多组 2 列数组
        k = 0
        for tgt in edge_list:
            for src in edge_list[tgt]:
                for rel in edge_list[tgt][src]:
                    pairs = np.array(edge_list[tgt][src][rel], dtype=np.int64)
                    ws    = np.array(edge_weight[tgt][src][rel], dtype=np.float32)
                    flat[f"EL_{k}_name"]  = np.array([f"{tgt}|{src}|{rel}"], dtype=object)
                    flat[f"EL_{k}_pairs"] = pairs
                    flat[f"EL_{k}_w"]     = ws
                    k += 1
        flat["EL_count"] = np.array([k], dtype=np.int64)

        np.savez_compressed(out_path, **flat)
        print(f"Saved to: {out_path}")

    return feature, times, edge_list, indxs, edge_weight


if __name__ == "__main__":
    build_graph_package()
