import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
import joblib
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

def carregar_dados_aluno(pasta_aluno: str) -> List[Dict]:
    """
    Carrega todos os arquivos JSON de performance de um aluno
    """
    dados = []
    arquivos = sorted([f for f in os.listdir(pasta_aluno) if f.endswith('.json')])
    
    for arquivo in arquivos:
        caminho = os.path.join(pasta_aluno, arquivo)
        with open(caminho, 'r', encoding='utf-8') as f:
            dados_dia = json.load(f)
            dados.append(dados_dia)
    
    return dados

def preparar_dados_treinamento(dados_histórico: List[Dict]) -> Tuple[pd.DataFrame, pd.Series, List[int], pd.DataFrame]:
    """
    Prepara dados para treinamento do ranker (usando a mesma lógica da API)
    """
    # Concatenar todos os dados históricos
    todos_dados = []
    for i, dados_dia in enumerate(dados_histórico):
        for item in dados_dia:
            item['data'] = i  # Adicionar dia como data
        todos_dados.extend(dados_dia)
    
    df = pd.DataFrame(todos_dados)
    
    # Calcular score como na API
    df["score"] = 1 - ((df["desempenho"] + 1) / 2)
    
    # Features como na API
    features = ["peso_classe", "peso_subclasse", "peso_por_questao"]
    
    X = df[features]
    y = df["score"]
    
    # Criar grupos por dia (como na API)
    group = df.groupby("data").size().tolist()
    
    return X, y, group, df

def treinar_ranker(X: pd.DataFrame, y: pd.Series, group: List[int]) -> object:
    """
    Treina o modelo XGBoost ranker (usando a mesma lógica da API)
    """
    try:
        from xgboost import XGBRanker
        
        # Configurar modelo de ranking como na API
        model = XGBRanker(
            objective="rank:pairwise",
            learning_rate=0.1,
            max_depth=4,
            n_estimators=100,
            verbosity=0  # silêncio interno do XGBoost
        )
        
        # Treinar modelo com grupos
        model.fit(X, y, group=group)
        return model
        
    except ImportError:
        print("❌ XGBoost não está instalado. Instalando...")
        os.system("pip install xgboost")
        from xgboost import XGBRanker
        
        model = XGBRanker(
            objective="rank:pairwise",
            learning_rate=0.1,
            max_depth=4,
            n_estimators=100,
            verbosity=0
        )
        
        model.fit(X, y, group=group)
        return model

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

def calcular_ndcg5(model: object, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Calcula NDCG@5 para o modelo treinado (usando a mesma lógica da API)
    """
    # Fazer predições
    predicoes = model.predict(X)
    
    # Calcular NDCG@5 usando a função da API
    return ndcg(y_true=y.values, y_pred=predicoes, k=5)

def obter_top5_conteudos(model: object, X: pd.DataFrame, df: pd.DataFrame) -> List[Dict]:
    """
    Retorna os 5 conteúdos com maior score de predição
    """
    predicoes = model.predict(X)
    
    # Adicionar predições ao DataFrame
    df_copy = df.copy()
    df_copy['predicao'] = predicoes
    
    # Agrupar por conteúdo e pegar o maior score de predição
    top_conteudos = df_copy.groupby(['classe', 'subclasse']).agg({
        'predicao': 'max',
        'desempenho': 'mean',
        'peso_classe': 'first',
        'peso_subclasse': 'first'
    }).reset_index()
    
    # Ordenar por predição e pegar top 5
    top5 = top_conteudos.nlargest(5, 'predicao')
    
    return top5[['classe', 'subclasse', 'desempenho', 'predicao']].to_dict('records')

def treinar_incremental(pasta_aluno: str, pasta_saida: str = "resultados_ranker"):
    """
    Treina o ranker incrementalmente e salva resultados
    """
    print(f"🎯 Treinamento incremental do ranker")
    print(f"📁 Pasta do aluno: {pasta_aluno}")
    
    # Carregar dados do aluno
    dados_aluno = carregar_dados_aluno(pasta_aluno)
    print(f"   📊 Total de dias carregados: {len(dados_aluno)}")
    
    # Criar pasta de saída
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Resultados
    resultados = []
    
    # Treinar incrementalmente
    for i in range(1, len(dados_aluno) + 1):
        print(f"\n📈 Treinamento {i}/{len(dados_aluno)} (usando {i} dias de dados)...")
        
        # Usar dados até o dia i
        dados_parciais = dados_aluno[:i]
        
        # Preparar dados
        X, y, group, df = preparar_dados_treinamento(dados_parciais)
        
        # Treinar modelo
        model = treinar_ranker(X, y, group)
        
        # Calcular NDCG@5
        ndcg5 = calcular_ndcg5(model, X, y)
        
        # Obter top 5 conteúdos
        top5_conteudos = obter_top5_conteudos(model, X, df)
        
        # Salvar resultado
        resultado = {
            'dia': i,
            'ndcg5': ndcg5,
            'top5_conteudos': top5_conteudos,
            'total_conteudos': len(df),
            'timestamp': datetime.now().isoformat()
        }
        
        resultados.append(resultado)
        
        print(f"   ✅ NDCG@5: {ndcg5:.4f}")
        print(f"   🎯 Top 1: {top5_conteudos[0]['classe']} - {top5_conteudos[0]['subclasse']}")
    
    # Salvar resultados
    nome_arquivo = f"resultado_ranker_{os.path.basename(pasta_aluno)}.json"
    caminho_saida = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Treinamento concluído!")
    print(f"   📁 Resultados salvos em: {caminho_saida}")
    print(f"   📊 Total de treinamentos: {len(resultados)}")
    
    # Mostrar resumo
    ndcg_final = resultados[-1]['ndcg5']
    print(f"   🏆 NDCG@5 final: {ndcg_final:.4f}")
    
    return resultados

def mostrar_resumo(resultados: List[Dict]):
    """
    Mostra resumo dos resultados
    """
    print(f"\n📊 Resumo dos resultados:")
    print(f"   📈 Evolução do NDCG@5:")
    
    for resultado in resultados[::max(1, len(resultados)//10)]:  # Mostrar a cada 10% dos dados
        print(f"      Dia {resultado['dia']:3d}: {resultado['ndcg5']:.4f}")
    
    # Top 5 final
    print(f"\n   🎯 Top 5 conteúdos finais:")
    top5_final = resultados[-1]['top5_conteudos']
    for i, conteudo in enumerate(top5_final, 1):
        print(f"      {i}. {conteudo['classe']} - {conteudo['subclasse']} (score: {conteudo['predicao']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar ranker incrementalmente')
    parser.add_argument('pasta_aluno', help='Pasta com os arquivos JSON de performance do aluno')
    parser.add_argument('--pasta-saida', default='resultados_ranker', 
                       help='Pasta para salvar resultados (padrão: resultados_ranker)')
    
    args = parser.parse_args()
    
    # Verificar se pasta existe
    if not os.path.exists(args.pasta_aluno):
        print(f"❌ Pasta não encontrada: {args.pasta_aluno}")
        exit(1)
    
    # Executar treinamento incremental
    resultados = treinar_incremental(args.pasta_aluno, args.pasta_saida)
    
    # Mostrar resumo
    mostrar_resumo(resultados) 