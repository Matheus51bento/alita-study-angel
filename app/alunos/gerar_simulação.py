import pandas as pd
import numpy as np
import os
import json
import random
from copy import deepcopy


def escolher_metrica_disp(df, eps=1e-6, thr_cv_irt=0.5, thr_cv_sqrt=0.2):
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
    std_p = df["p_norm"].std()
    cv = std_p / (mean_p + eps)  # coef. de variação

    # 3) clipping para IRT
    df["p_clip"] = df["p_norm"].clip(eps, 1 - eps)

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
        df["lacuna"] * df["peso_classe"] * df["peso_subclasse"] * df["peso_por_questao"]
    )

    # informar dispersão
    return df


def gerar_simulacao_de_desempenho(df, student_id, dias=200, resultados=None):
    if resultados is None:
        resultados = []

    if dias == 0:
        return resultados

    df = escolher_metrica_disp(df)

    top_5_materias = df.sort_values("prioridade", ascending=False).head(5)

    for index, row in top_5_materias.iterrows():
        taxa_melhoria = random.uniform(0.01, 0.1)  # Sempre positivo

        # Atualiza o desempenho (limitado a 1.0)
        novo_desempenho = min(1.0, row["desempenho"] + taxa_melhoria)

        df.at[index, "desempenho"] = novo_desempenho

    # Adiciona o student_id aos resultados de cada dia
    resultado_dia = df[
        [
            "classe",
            "subclasse",
            "desempenho",
            "peso_classe",
            "peso_subclasse",
            "peso_por_questao",
        ]
    ].to_dict(orient="records")

    # Adiciona o student_id no resultado
    for item in resultado_dia:
        item["student_id"] = str(student_id)

    resultados.append(deepcopy(resultado_dia))

    return gerar_simulacao_de_desempenho(df, student_id, dias - 1, resultados)


def gerar_arquivos_de_desempenho(dados, dias=200, aluno="aluno"):
    # Cria a pasta para o aluno, se não existir
    pasta_aluno = os.path.join("output", aluno)
    if not os.path.exists(pasta_aluno):
        os.makedirs(pasta_aluno)

    for i in range(dias):
        # Cria o nome do arquivo para o dia, dentro da pasta do aluno
        arquivo_nome = os.path.join(pasta_aluno, f"desempenho_dia_{i+1}.json")

        # Salva o arquivo JSON na pasta do aluno
        with open(arquivo_nome, "w") as f:
            json.dump(dados[i], f, indent=4)
        print(f"Arquivo gerado: {arquivo_nome}")


df = pd.read_csv("alunos/aluno2.csv")

# Gerar simulação de 200 dias de desempenho
dados_simulados = gerar_simulacao_de_desempenho(df, 8000, dias=200)

# Gerar arquivos JSON
gerar_arquivos_de_desempenho(dados_simulados, dias=200)
