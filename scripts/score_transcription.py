import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Remove bracket numbers
def remove_brackets(text):
    return re.sub(r"\s*\(.*?\)", "", str(text)).strip()

# Compute similarity
def compute_similarity(target, response):
    emb1 = model.encode(target)
    emb2 = model.encode(response)
    return util.cos_sim(emb1, emb2).item()

# Convert similarity to score
def assign_score(sim):
    if sim >= 0.85:
        return 2
    elif sim >= 0.65:
        return 1
    else:
        return 0

# Load all participant CSV files
data_folder = r'data/transcription'
all_dfs = []

for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path)
        df["participant"] = file
        all_dfs.append(df.iloc[0:30,::])

data = pd.concat(all_dfs, ignore_index=True)

# Clean stimulus
data["Stimulus"] = data["Stimulus"].apply(remove_brackets)

# Compute similarity
data["Similarity"] = data.apply(
    lambda row: compute_similarity(row["Stimulus"], row["Transcription Rater 1"]),
    axis=1
)

# Assign scores
data["Predicted_score"] = data["Similarity"].apply(assign_score)

# Save results
output_path = "results/AutoEIT_scored_output.csv"
data.to_csv(output_path, index=False)

print("Scoring complete. Results saved to:", output_path)