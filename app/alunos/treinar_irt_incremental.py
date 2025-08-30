import pandas as pd
import numpy as np
import os
import json
import random
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

    return df

def dcg(relevancias, k):
    """
    Calcula DCG (Discounted Cumulative Gain) - função da API
    """
    relevancias = np.asarray(relevancias)[:k]
    if relevancias.size:
        return relevancias[0] + np.sum(relevancias[1:] / np.log2(np.arange(2, relevancias.size + 1)))
    return 0.0

def ndcg(y_true, y_pred, k=5):
    """
    Calcula NDCG (Normalized Discounted Cumulative Gain) - função da API
    """
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order)
    dcg_max = dcg(sorted(y_true, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg(y_true_sorted, k) / dcg_max

def calcular_ndcg5_irt(df: pd.DataFrame) -> float:
    """
    Calcula NDCG@5 para o modelo IRT
    """
    # Usar prioridade como score de predição
    y_pred = df['prioridade'].values
    
    # Usar score de relevância baseado no desempenho e pesos
    y_true = (1 - ((df["desempenho"] + 1) / 2)) * df["peso_classe"] * df["peso_subclasse"] * df["peso_por_questao"]
    
    return ndcg(y_true=y_true.values, y_pred=y_pred, k=5)

def obter_top5_conteudos_irt(df: pd.DataFrame) -> List[Dict]:
    """
    Retorna os 5 conteúdos com maior prioridade IRT
    """
    # Ordenar por prioridade e pegar top 5
    top5 = df.nlargest(5, 'prioridade')
    
    return top5[['classe', 'subclasse', 'desempenho', 'prioridade', 'metrica']].to_dict('records')

def simular_irt_incremental(df_inicial: pd.DataFrame, dias: int = 200) -> List[Dict]:
    """
    Simula o processo IRT incrementalmente dia a dia
    """
    print(f"🎯 Simulação IRT incremental")
    print(f"   📊 Dias a simular: {dias}")
    
    df = df_inicial.copy()
    resultados = []
    
    for dia in range(1, dias + 1):
        print(f"\n📈 Dia {dia}/{dias}...")
        
        # Aplicar métrica IRT
        df = escolher_metrica_disp(df)
        
        # Calcular NDCG@5
        ndcg5 = calcular_ndcg5_irt(df)
        
        # Obter top 5 conteúdos
        top5_conteudos = obter_top5_conteudos_irt(df)
        
        # Salvar resultado do dia
        resultado = {
            'dia': dia,
            'ndcg5': ndcg5,
            'top5_conteudos': top5_conteudos,
            'total_conteudos': len(df),
            'metrica_utilizada': df['metrica'].iloc[0],  # Métrica mais usada
            'timestamp': datetime.now().isoformat()
        }
        
        resultados.append(resultado)
        
        print(f"   ✅ NDCG@5: {ndcg5:.4f}")
        print(f"   🎯 Top 1: {top5_conteudos[0]['classe']} - {top5_conteudos[0]['subclasse']}")
        print(f"   📊 Métrica: {top5_conteudos[0]['metrica']}")
        
        # Simular melhoria nos top 5 conteúdos (como no gerar_simulação.py)
        top_5_materias = df.sort_values("prioridade", ascending=False).head(5)
        
        for index, row in top_5_materias.iterrows():
            taxa_melhoria = random.uniform(0.01, 0.1)  # Sempre positivo
            
            # Atualiza o desempenho (limitado a 1.0)
            novo_desempenho = min(1.0, row["desempenho"] + taxa_melhoria)
            df.at[index, "desempenho"] = novo_desempenho
    
    return resultados

def carregar_dados_aluno(arquivo_csv: str) -> pd.DataFrame:
    """
    Carrega dados do aluno de um arquivo CSV
    """
    if not os.path.exists(arquivo_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_csv}")
    
    df = pd.read_csv(arquivo_csv)
    print(f"📚 Dados carregados: {len(df)} conteúdos")
    
    return df

def salvar_resultados(resultados: List[Dict], pasta_saida: str, nome_aluno: str):
    """
    Salva os resultados em arquivo JSON
    """
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    nome_arquivo = f"resultado_irt_{nome_aluno}.json"
    caminho_saida = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Resultados salvos em: {caminho_saida}")
    
    return caminho_saida

def mostrar_resumo(resultados: List[Dict]):
    """
    Mostra resumo dos resultados
    """
    print(f"\n📊 Resumo dos resultados IRT:")
    print(f"   📈 Evolução do NDCG@5:")
    
    for resultado in resultados[::max(1, len(resultados)//10)]:  # Mostrar a cada 10% dos dados
        print(f"      Dia {resultado['dia']:3d}: {resultado['ndcg5']:.4f} ({resultado['metrica_utilizada']})")
    
    # Top 5 final
    print(f"\n   🎯 Top 5 conteúdos finais:")
    top5_final = resultados[-1]['top5_conteudos']
    for i, conteudo in enumerate(top5_final, 1):
        print(f"      {i}. {conteudo['classe']} - {conteudo['subclasse']} (prioridade: {conteudo['prioridade']:.4f}, métrica: {conteudo['metrica']})")

def analisar_metricas(resultados: List[Dict]):
    """
    Analisa quais métricas foram mais utilizadas
    """
    metricas = [r['metrica_utilizada'] for r in resultados]
    metricas_unicas = list(set(metricas))
    
    print(f"\n🔍 Análise de métricas utilizadas:")
    for metrica in metricas_unicas:
        count = metricas.count(metrica)
        percentual = (count / len(metricas)) * 100
        print(f"   📊 {metrica}: {count} dias ({percentual:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simular IRT incrementalmente')
    parser.add_argument('arquivo_csv', help='Caminho do arquivo CSV do aluno')
    parser.add_argument('--dias', type=int, default=200, help='Número de dias para simular (padrão: 200)')
    parser.add_argument('--pasta-saida', default='resultados_irt', 
                       help='Pasta para salvar resultados (padrão: resultados_irt)')
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    if not os.path.exists(args.arquivo_csv):
        print(f"❌ Arquivo não encontrado: {args.arquivo_csv}")
        exit(1)
    
    # Carregar dados do aluno
    df_aluno = carregar_dados_aluno(args.arquivo_csv)
    
    # Simular IRT incremental
    resultados = simular_irt_incremental(df_aluno, dias=args.dias)
    
    # Salvar resultados
    nome_aluno = os.path.splitext(os.path.basename(args.arquivo_csv))[0]
    salvar_resultados(resultados, args.pasta_saida, nome_aluno)
    
    # Mostrar resumo
    mostrar_resumo(resultados)
    
    # Analisar métricas
    analisar_metricas(resultados)
    
    print(f"\n🎉 Simulação IRT concluída!")
    print(f"   📊 Total de dias simulados: {len(resultados)}")
    print(f"   🏆 NDCG@5 final: {resultados[-1]['ndcg5']:.4f}") 