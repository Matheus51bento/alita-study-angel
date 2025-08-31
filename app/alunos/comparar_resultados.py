import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
import seaborn as sns

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def carregar_resultados_ranker(pasta_resultados: str) -> Dict[int, List[Dict]]:
    """
    Carrega todos os resultados do ranker
    """
    resultados = {}
    
    if not os.path.exists(pasta_resultados):
        print(f"❌ Pasta de resultados do ranker não encontrada: {pasta_resultados}")
        return resultados
    
    for arquivo in os.listdir(pasta_resultados):
        if arquivo.startswith("resultado_ranker_aluno_") and arquivo.endswith(".json"):
            # Extrair ID do aluno do nome do arquivo
            student_id = int(arquivo.split("_")[-1].replace(".json", ""))
            
            caminho = os.path.join(pasta_resultados, arquivo)
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                resultados[student_id] = dados
    
    print(f"📊 Carregados {len(resultados)} resultados do ranker")
    return resultados

def carregar_resultados_irt(pasta_resultados: str) -> Dict[int, List[Dict]]:
    """
    Carrega todos os resultados do IRT
    """
    resultados = {}
    
    if not os.path.exists(pasta_resultados):
        print(f"❌ Pasta de resultados do IRT não encontrada: {pasta_resultados}")
        return resultados
    
    for arquivo in os.listdir(pasta_resultados):
        if arquivo.startswith("resultado_irt_aluno_") and arquivo.endswith(".json"):
            # Extrair ID do aluno do nome do arquivo
            student_id = int(arquivo.split("_")[-1].replace(".json", ""))
            
            caminho = os.path.join(pasta_resultados, arquivo)
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                resultados[student_id] = dados
    
    print(f"📊 Carregados {len(resultados)} resultados do IRT")
    return resultados

def criar_dataframe_comparativo(resultados_ranker: Dict, resultados_irt: Dict) -> pd.DataFrame:
    """
    Cria DataFrame com dados comparativos
    """
    dados = []
    
    # Processar dados do ranker
    for student_id, dados_ranker in resultados_ranker.items():
        for dia_dados in dados_ranker:
            dados.append({
                'student_id': student_id,
                'dia': dia_dados['dia'],
                'ndcg5_ranker': dia_dados['ndcg5'],
                'ndcg5_irt': None,  # Será preenchido depois
                'modelo': 'ranker'
            })
    
    # Processar dados do IRT
    for student_id, dados_irt in resultados_irt.items():
        for dia_dados in dados_irt:
            dados.append({
                'student_id': student_id,
                'dia': dia_dados['dia'],
                'ndcg5_ranker': None,  # Será preenchido depois
                'ndcg5_irt': dia_dados['ndcg5'],
                'modelo': 'irt'
            })
    
    df = pd.DataFrame(dados)
    
    # Preencher valores faltantes
    for student_id in df['student_id'].unique():
        for dia in df[df['student_id'] == student_id]['dia'].unique():
            # Encontrar valores do ranker
            ranker_val = df[(df['student_id'] == student_id) & 
                           (df['dia'] == dia) & 
                           (df['modelo'] == 'ranker')]['ndcg5_ranker'].iloc[0] if len(df[(df['student_id'] == student_id) & (df['dia'] == dia) & (df['modelo'] == 'ranker')]) > 0 else None
            
            # Encontrar valores do IRT
            irt_val = df[(df['student_id'] == student_id) & 
                        (df['dia'] == dia) & 
                        (df['modelo'] == 'irt')]['ndcg5_irt'].iloc[0] if len(df[(df['student_id'] == student_id) & (df['dia'] == dia) & (df['modelo'] == 'irt')]) > 0 else None
            
            # Atualizar valores
            df.loc[(df['student_id'] == student_id) & (df['dia'] == dia), 'ndcg5_ranker'] = ranker_val
            df.loc[(df['student_id'] == student_id) & (df['dia'] == dia), 'ndcg5_irt'] = irt_val
    
    return df

def plotar_evolucao_individual(df: pd.DataFrame, student_id: int, pasta_saida: str = "graficos"):
    """
    Plota evolução individual de um aluno
    """
    dados_aluno = df[df['student_id'] == student_id].copy()
    
    if dados_aluno.empty:
        print(f"⚠️  Dados não encontrados para aluno {student_id}")
        return
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Evolução temporal
    dias = dados_aluno['dia'].unique()
    dias.sort()
    
    ndcg_ranker = [dados_aluno[dados_aluno['dia'] == dia]['ndcg5_ranker'].iloc[0] for dia in dias]
    ndcg_irt = [dados_aluno[dados_aluno['dia'] == dia]['ndcg5_irt'].iloc[0] for dia in dias]
    
    ax1.plot(dias, ndcg_ranker, 'o-', label='XGBoost Ranker', linewidth=2, markersize=4)
    ax1.plot(dias, ndcg_irt, 's-', label='IRT', linewidth=2, markersize=4)
    ax1.set_xlabel('Dia')
    ax1.set_ylabel('NDCG@5')
    ax1.set_title(f'Evolução NDCG@5 - Aluno {student_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparação final
    modelos = ['XGBoost Ranker', 'IRT']
    valores_finais = [ndcg_ranker[-1], ndcg_irt[-1]]
    cores = ['#2E86AB', '#A23B72']
    
    bars = ax2.bar(modelos, valores_finais, color=cores, alpha=0.7)
    ax2.set_ylabel('NDCG@5 Final')
    ax2.set_title(f'Comparação Final - Aluno {student_id}')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores_finais):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    nome_arquivo = f"evolucao_aluno_{student_id}.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico salvo: {caminho}")

def plotar_comparacao_geral(df: pd.DataFrame, pasta_saida: str = "graficos"):
    """
    Plota comparação geral de todos os alunos
    """
    # Calcular estatísticas por aluno
    stats_alunos = []
    
    for student_id in df['student_id'].unique():
        dados_aluno = df[df['student_id'] == student_id]
        
        # Valores finais
        ndcg_ranker_final = dados_aluno['ndcg5_ranker'].iloc[-1]
        ndcg_irt_final = dados_aluno['ndcg5_irt'].iloc[-1]
        
        # Médias
        ndcg_ranker_media = dados_aluno['ndcg5_ranker'].mean()
        ndcg_irt_media = dados_aluno['ndcg5_irt'].mean()
        
        stats_alunos.append({
            'student_id': student_id,
            'ndcg_ranker_final': ndcg_ranker_final,
            'ndcg_irt_final': ndcg_irt_final,
            'ndcg_ranker_media': ndcg_ranker_media,
            'ndcg_irt_media': ndcg_irt_media,
            'diferenca': ndcg_ranker_final - ndcg_irt_final
        })
    
    stats_df = pd.DataFrame(stats_alunos)
    
    # Criar figura com múltiplos subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Comparação final por aluno
    x_pos = np.arange(len(stats_df))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, stats_df['ndcg_ranker_final'], width, 
                    label='XGBoost Ranker', alpha=0.7, color='#2E86AB')
    bars2 = ax1.bar(x_pos + width/2, stats_df['ndcg_irt_final'], width, 
                    label='IRT', alpha=0.7, color='#A23B72')
    
    ax1.set_xlabel('Aluno')
    ax1.set_ylabel('NDCG@5 Final')
    ax1.set_title('Comparação Final por Aluno')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Aluno {id}' for id in stats_df['student_id']], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Diferença entre modelos
    cores_diferenca = ['green' if x > 0 else 'red' for x in stats_df['diferenca']]
    bars = ax2.bar(x_pos, stats_df['diferenca'], color=cores_diferenca, alpha=0.7)
    ax2.set_xlabel('Aluno')
    ax2.set_ylabel('Diferença (Ranker - IRT)')
    ax2.set_title('Diferença entre Modelos')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Aluno {id}' for id in stats_df['student_id']], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, stats_df['diferenca']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if valor > 0 else -0.01), 
                f'{valor:.3f}', ha='center', va='bottom' if valor > 0 else 'top', fontweight='bold')
    
    # Gráfico 3: Boxplot comparativo
    dados_boxplot = []
    labels = []
    
    for student_id in df['student_id'].unique():
        dados_aluno = df[df['student_id'] == student_id]
        dados_boxplot.extend([dados_aluno['ndcg5_ranker'].values, dados_aluno['ndcg5_irt'].values])
        labels.extend([f'Aluno {student_id}\nRanker', f'Aluno {student_id}\nIRT'])
    
    box_plot = ax3.boxplot(dados_boxplot, labels=labels, patch_artist=True)
    
    # Colorir boxes
    cores = ['#2E86AB', '#A23B72'] * len(df['student_id'].unique())
    for patch, color in zip(box_plot['boxes'], cores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('NDCG@5')
    ax3.set_title('Distribuição NDCG@5 por Aluno e Modelo')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Estatísticas gerais
    modelos = ['XGBoost Ranker', 'IRT']
    media_final = [stats_df['ndcg_ranker_final'].mean(), stats_df['ndcg_irt_final'].mean()]
    std_final = [stats_df['ndcg_ranker_final'].std(), stats_df['ndcg_irt_final'].std()]
    
    bars = ax4.bar(modelos, media_final, yerr=std_final, capsize=5, 
                   color=['#2E86AB', '#A23B72'], alpha=0.7)
    ax4.set_ylabel('NDCG@5 Médio')
    ax4.set_title('Estatísticas Gerais')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, media_final):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    nome_arquivo = "comparacao_geral.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico geral salvo: {caminho}")
    
    return stats_df

def gerar_relatorio(stats_df: pd.DataFrame, pasta_saida: str = "graficos"):
    """
    Gera relatório com estatísticas
    """
    relatorio = f"""
# Relatório de Comparação: XGBoost Ranker vs IRT

## Estatísticas Gerais

### Médias Finais:
- **XGBoost Ranker**: {stats_df['ndcg_ranker_final'].mean():.4f} ± {stats_df['ndcg_ranker_final'].std():.4f}
- **IRT**: {stats_df['ndcg_irt_final'].mean():.4f} ± {stats_df['ndcg_irt_final'].std():.4f}

### Diferença Média:
- **Ranker - IRT**: {stats_df['diferenca'].mean():.4f} ± {stats_df['diferenca'].std():.4f}

### Análise por Aluno:
"""
    
    for _, row in stats_df.iterrows():
        melhor_modelo = "Ranker" if row['diferenca'] > 0 else "IRT"
        relatorio += f"""
**Aluno {row['student_id']}**:
- Ranker: {row['ndcg_ranker_final']:.4f}
- IRT: {row['ndcg_irt_final']:.4f}
- Diferença: {row['diferenca']:.4f}
- Melhor: {melhor_modelo}
"""
    
    # Salvar relatório
    nome_arquivo = "relatorio_comparacao.md"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    print(f"📄 Relatório salvo: {caminho}")

def main():
    parser = argparse.ArgumentParser(description='Comparar resultados do ranker e IRT')
    parser.add_argument('--pasta-ranker', default='resultados_ranker_teste', 
                       help='Pasta com resultados do ranker')
    parser.add_argument('--pasta-irt', default='resultados_irt_teste', 
                       help='Pasta com resultados do IRT')
    parser.add_argument('--pasta-saida', default='graficos', 
                       help='Pasta para salvar gráficos')
    parser.add_argument('--alunos', nargs='+', type=int, 
                       help='IDs específicos dos alunos para analisar (opcional)')
    
    args = parser.parse_args()
    
    print("🔍 Carregando resultados...")
    
    # Carregar dados
    resultados_ranker = carregar_resultados_ranker(args.pasta_ranker)
    resultados_irt = carregar_resultados_irt(args.pasta_irt)
    
    if not resultados_ranker or not resultados_irt:
        print("❌ Não foi possível carregar os resultados!")
        return
    
    # Criar DataFrame comparativo
    df = criar_dataframe_comparativo(resultados_ranker, resultados_irt)
    
    if df.empty:
        print("❌ Nenhum dado encontrado para comparação!")
        return
    
    print(f"📊 Dados carregados: {len(df)} registros")
    
    # Gerar gráficos individuais
    alunos_para_analisar = args.alunos if args.alunos else df['student_id'].unique()
    
    print(f"\n📈 Gerando gráficos individuais...")
    for student_id in alunos_para_analisar:
        plotar_evolucao_individual(df, student_id, args.pasta_saida)
    
    # Gerar gráfico geral
    print(f"\n📊 Gerando gráfico comparativo geral...")
    stats_df = plotar_comparacao_geral(df, args.pasta_saida)
    
    # Gerar relatório
    print(f"\n📄 Gerando relatório...")
    gerar_relatorio(stats_df, args.pasta_saida)
    
    print(f"\n🎉 Análise concluída!")
    print(f"📁 Gráficos salvos em: {args.pasta_saida}")

if __name__ == "__main__":
    main() 