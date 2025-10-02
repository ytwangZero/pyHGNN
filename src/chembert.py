import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# 1. 加载 CSV
df = pd.read_csv("data/node_smiles_all.csv")
smiles_list = df["smiles"].tolist()

# 2. 加载 ChemBERTa 模型并移至GPU（如果可用）
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"Using device: {device}")

# 3. 批量提取 embedding
batch_size = 32
embeddings = []

for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing batches"):
    batch = smiles_list[i:i+batch_size]
    
    try:
        # 批量编码（动态padding）
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,  # 动态padding到batch内最大长度
            max_length=128
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        embeddings.extend(emb)
        
    except Exception as e:
        print(f"\nError processing batch {i//batch_size}: {e}")
        # 填充零向量
        embeddings.extend([np.zeros(768)] * len(batch))

# 4. 保存结果
embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

emb_df = pd.DataFrame(
    embeddings,
    columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
)
df = pd.concat([df, emb_df], axis=1)
df.to_csv("result/smiles_emb.csv", index=False)

print("Embeddings saved successfully!")