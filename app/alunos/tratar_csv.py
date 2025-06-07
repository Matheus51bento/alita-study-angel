import pandas as pd

df = pd.read_csv("aluno2.csv")

df["peso_por_questao"] = df["peso_por_questao"].str.replace(",", ".").astype(float)
df["peso_subclasse"] = df["peso_subclasse"].str.replace(",", ".").astype(float)

df.to_csv("aluno2.csv", index=False)