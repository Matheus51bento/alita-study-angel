import pandas as pd
import numpy as np
import os
import json
import random
import argparse
from copy import deepcopy


def escolher_metrica_disp(df, eps=1e-6, thr_cv_irt=0.5, thr_cv_sqrt=0.2):
    """
    Escolhe entre:
      - IRT 2PL, se coeficiente de variação alto
      - sqrt, se coeficiente moderado
      - linear, se coeficiente baixo
    """
    df = df.copy()
    df["p_norm"] = (df["desempenho"] + 1) / 2
    mean_p = df["p_norm"].mean()
    std_p = df["p_norm"].std()
    cv = std_p / (mean_p + eps)

    df["p_clip"] = df["p_norm"].clip(eps, 1 - eps)

    if cv >= thr_cv_irt:
        df["lacuna"] = -np.log(df["p_clip"] / (1 - df["p_clip"]))
        df["metrica"] = "IRT 2PL"
    elif cv >= thr_cv_sqrt:
        df["lacuna"] = np.sqrt(1 - df["p_norm"])
        df["metrica"] = "sqrt(1-p)"
    else:
        df["lacuna"] = 1 - df["p_norm"]
        df["metrica"] = "linear(1-p)"

    df["prioridade"] = (
        df["lacuna"] * df["peso_classe"] * df["peso_subclasse"] * df["peso_por_questao"]
    )

    return df


def gerar_simulacao_de_desempenho(df, student_id, dias=200, resultados=None):
    if resultados is None:
        resultados = []

    if dias == 0:
        return resultados

    df = escolher_metrica_disp(df)

    top_5_materias = df.sort_values("prioridade", ascending=False).head(5)

    for index, row in top_5_materias.iterrows():
        taxa_melhoria = random.uniform(0.01, 0.1)

        novo_desempenho = min(1.0, row["desempenho"] + taxa_melhoria)

        df.at[index, "desempenho"] = novo_desempenho

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

    for item in resultado_dia:
        item["student_id"] = str(student_id)

    resultados.append(deepcopy(resultado_dia))

    return gerar_simulacao_de_desempenho(df, student_id, dias - 1, resultados)


def gerar_arquivos_de_desempenho(dados, dias=200, aluno="aluno"):
    pasta_aluno = os.path.join("output", aluno)
    if not os.path.exists(pasta_aluno):
        os.makedirs(pasta_aluno)

    for i in range(dias):
        arquivo_nome = os.path.join(pasta_aluno, f"desempenho_dia_{i+1}.json")

        with open(arquivo_nome, "w") as f:
            json.dump(dados[i], f, indent=4)
        print(f"Arquivo gerado: {arquivo_nome}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gerar simulação de desempenho para um aluno')
    parser.add_argument('arquivo_csv', help='Caminho do arquivo CSV do aluno')
    parser.add_argument('student_id', type=int, help='ID do aluno')
    parser.add_argument('--dias', type=int, default=200, help='Número de dias para simular (padrão: 200)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.arquivo_csv):
        print(f"❌ Arquivo não encontrado: {args.arquivo_csv}")
        exit(1)
    
    print(f"📚 Carregando arquivo: {args.arquivo_csv}")
    df = pd.read_csv(args.arquivo_csv)
    print(f"   Conteúdos carregados: {len(df)}")
    
    print(f"🎯 Gerando simulação para aluno {args.student_id} por {args.dias} dias...")
    
    dados_simulados = gerar_simulacao_de_desempenho(df, args.student_id, dias=args.dias)
    
    print(f"💾 Salvando arquivos...")
    
    gerar_arquivos_de_desempenho(dados_simulados, dias=args.dias, aluno=f"aluno_{args.student_id}")
    
    print(f"✅ Simulação concluída!")
    print(f"   📁 Arquivos salvos em: output/aluno_{args.student_id}/")
    print(f"   📊 Total de dias simulados: {args.dias}") 