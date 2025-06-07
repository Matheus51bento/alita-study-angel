import pandas as pd
import numpy as np

def escolher_metrica_disp(df,
                          eps=1e-6,
                          thr_cv_irt=0.5,
                          thr_cv_sqrt=0.2):
    """
    Escolhe entre:
      - IRT 2PL, se coeficiente de variação alto
      - sqrt, se coeficiente moderado
      - linear, se coeficiente baixo
    """
    df = df.copy()
    # 1) normaliza desempenho [-1,1] → [0,1]
    df["p_norm"] = (df["desempenho"] + 1) / 2
    # 2) estatísticas de dispersão
    mean_p = df["p_norm"].mean()
    std_p  = df["p_norm"].std()
    cv     = std_p / (mean_p + eps)    # coef. de variação

    # 3) clipping para IRT
    df["p_clip"] = df["p_norm"].clip(eps, 1-eps)

    # 4) escolhe lacuna
    if cv >= thr_cv_irt:
        df["lacuna"] = -np.log(df["p_clip"] / (1 - df["p_clip"]))
        df["metrica"] = "IRT 2PL"
    elif cv >= thr_cv_sqrt:
        df["lacuna"] = np.sqrt(1 - df["p_norm"])
        df["metrica"] = "sqrt(1-p)"
    else:
        df["lacuna"] = 1 - df["p_norm"]
        df["metrica"] = "linear(1-p)"

    # 5) prioridade combinando todos os pesos
    df["prioridade"] = (
        df["lacuna"]
      * df["peso_classe"]
      * df["peso_subclasse"]
      * df["peso_por_questao"]
    )

    # informar dispersão
    print(f"μ(p_norm) = {mean_p:.3f}, σ = {std_p:.3f}, CV = {cv:.3f}")
    return df

# df = pd.read_csv("aluno2.csv")
# res = escolher_metrica_disp(df)

# print("\nTop 10 conteúdos a revisar:")
# print(res.sort_values("prioridade", ascending=False)[
#     ["classe","subclasse","p_norm","lacuna","prioridade","metrica"]
# ].head(10).to_string(index=False))