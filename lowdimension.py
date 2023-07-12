import pandas as pd
from umap.parametric_umap import ParametricUMAP
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diamonds.csv")
for a in ['cut', 'color', 'clarity']:
    df = df.drop(a, axis=1)

df_scaled = StandardScaler().fit_transform(df)
embedding = ParametricUMAP(n_epochs=50, verbose=True, n_jobs=2, low_memory=True, n_components=2).fit(df_scaled)
#pd.DataFrame(embedding.embedding_).to_csv("dataset_paradas_congalsa_copula_embedding.csv", index=False)
print(embedding.embedding_.shape)
