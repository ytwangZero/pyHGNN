import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
import torch_geometric.transforms as T

# 路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

# =====================
# 1. 读取节点和边 CSV
# =====================
node_f = pd.read_csv(os.path.join(DATA_DIR, "Node_F.csv"))
node_m = pd.read_csv(os.path.join(DATA_DIR, "Node_M.csv"))
edge_fm = pd.read_csv(os.path.join(DATA_DIR, "Edge_FM.csv"))
edge_mm = pd.read_csv(os.path.join(DATA_DIR, "Edge_MM.csv"))

# 建立 ID → 索引映射
feat_ids = pd.Index(node_f["ID"].astype(str))
met_ids = pd.Index(node_m["ID"].astype(str))
feat2idx = {k: i for i, k in enumerate(feat_ids)}
met2idx = {k: i for i, k in enumerate(met_ids)}

# =====================
# 2. 构造 HeteroData
# =====================
data = HeteroData()

# 节点特征
data["feature"].x = torch.tensor(node_f[["weight_p","weight_fc"]].values,
                                 dtype=torch.float32)
data["metabolite"].x = torch.tensor(node_m[["weight_d","weight_b"]].values,
                                    dtype=torch.float32)

# Feature–Metabolite edges
fm = edge_fm[edge_fm["from"].isin(feat_ids) & edge_fm["to"].isin(met_ids)]
src = torch.tensor(fm["from"].map(feat2idx).values, dtype=torch.long)
dst = torch.tensor(fm["to"].map(met2idx).values, dtype=torch.long)
data["feature", "annotates", "metabolite"].edge_index = torch.stack([src, dst])

# Metabolite–Metabolite edges
mm = edge_mm[edge_mm["from"].isin(met_ids) & edge_mm["to"].isin(met_ids)]
m_src = torch.tensor(mm["from"].map(met2idx).values, dtype=torch.long)
m_dst = torch.tensor(mm["to"].map(met2idx).values, dtype=torch.long)
data["metabolite", "reacts_with", "metabolite"].edge_index = torch.stack([m_src, m_dst])

# 转无向，自动加反向边
data = T.ToUndirected()(data)

print(data)
print("Edge types:", data.edge_types)

# =====================
# 3. 定义模型
# =====================
class HGTEncoder(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=32, out_dim=32, heads=2):
        super().__init__()
        self.proj = nn.ModuleDict()
        for ntype in metadata[0]:
            self.proj[ntype] = nn.Linear(in_dims[ntype], hidden_dim)

        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
        self.conv2 = HGTConv(hidden_dim, out_dim, metadata, heads=heads)

    def forward(self, x_dict, edge_index_dict):
        # 投影到同一维度
        x_dict = {k: F.relu(self.proj[k](x)) for k, x in x_dict.items()}
        # 两层 HGTConv
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

metadata = data.metadata()
in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}

model = HGTEncoder(metadata, in_dims, hidden_dim=64, out_dim=64).to("cpu")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# =====================
# 4. 无监督训练 (link prediction)
# =====================
def negative_sampling(ei, num_src, num_dst, num_neg=None):
    num_neg = num_neg or ei.size(1)
    u = torch.randint(0, num_src, (num_neg,))
    v = torch.randint(0, num_dst, (num_neg,))
    return u, v

def dot_score(z_src, z_dst, u, v):
    return (z_src[u] * z_dst[v]).sum(dim=-1)

bce = nn.BCEWithLogitsLoss()

for epoch in range(1, 51):
    model.train()
    opt.zero_grad()
    z = model(data.x_dict, data.edge_index_dict)

    loss = 0
    for et in data.edge_types:
        src_type, _, dst_type = et
        ei = data[et].edge_index
        pos_u, pos_v = ei[0], ei[1]
        neg_u, neg_v = negative_sampling(ei, data[src_type].num_nodes, data[dst_type].num_nodes)

        pos_logit = dot_score(z[src_type], z[dst_type], pos_u, pos_v)
        neg_logit = dot_score(z[src_type], z[dst_type], neg_u, neg_v)
        y = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        logit = torch.cat([pos_logit, neg_logit])

        loss += bce(logit, y)

    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# =====================
# 5. 保存节点嵌入
# =====================
model.eval()
with torch.no_grad():
    z = model(data.x_dict, data.edge_index_dict)

embeddings = {}
for ntype in z:
    embeddings[ntype] = z[ntype].cpu().numpy()

# 保存 metabolite embedding
met_ids = node_m["ID"].tolist()
met_emb = pd.DataFrame(embeddings["metabolite"], index=met_ids)
met_emb.to_csv(os.path.join(ART_DIR, "metabolite_embedding.csv"))
print("Saved metabolite embedding to artifacts/metabolite_embedding.csv")

# 保存 feature embedding
feat_ids = node_f["ID"].tolist()
feat_emb = pd.DataFrame(embeddings["feature"], index=feat_ids)
feat_emb.to_csv(os.path.join(ART_DIR, "feature_embedding.csv"))
print("Saved feature embedding to artifacts/feature_embedding.csv")
