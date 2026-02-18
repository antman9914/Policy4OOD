import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load state index mapping
state_dict, state_map = {}, {}
state_list = pd.read_csv('processed_data/state_ref.csv')
state_name, state_abbr = state_list['state_name'], state_list['state_abbr']
idx = 0
for i in range(len(state_name)):
    if state_abbr[i] != 'HI' and state_abbr[i] != 'AK':
        state_dict[state_name[i]] = state_abbr[i]
        state_map[state_abbr[i]] = idx
        idx += 1
state_name, state_abbr = list(state_dict.keys()), list(state_map.keys())

# Customize this part according to your demand
df = pd.read_parquet('your_path/output/relationships.parquet')

# -------- clean --------
df["source"] = df["source"].astype(str)
df["target"] = df["target"].astype(str)
df["description"] = df["description"].fillna("").astype(str)

# -------- node indexing --------
nodes = pd.Index(pd.unique(pd.concat([df["source"], df["target"]])))
node2id = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

src = df["source"].map(node2id).to_numpy(dtype=np.int64)
dst = df["target"].map(node2id).to_numpy(dtype=np.int64)

# -------- edge_index --------
edge_index = np.stack([src, dst], axis=0)  # [2, E]
E = edge_index.shape[1]

# -------- node features (degree-based, no adjacency matrix) --------
node_text = list(node2id.keys())
edge_text = df['description']
model = SentenceTransformer("all-MiniLM-L6-v2")

node_embeddings = model.encode(node_text)
edge_embeddings = model.encode(edge_text)

# Customize this part according to your demand
kg_path = "your_kg_path"
if not os.path.exists(kg_path):
    os.makedirs(kg_path)
np.save(kg_path + "adj.npy", edge_index)
np.save(kg_path + "node.npy", node_embeddings)
np.save(kg_path + "edge.npy", edge_embeddings)