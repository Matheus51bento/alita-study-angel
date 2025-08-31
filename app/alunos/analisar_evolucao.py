import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
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

def analisar_evolucao_individual(dados_aluno: List[Dict], student_id: int, pasta_saida: str = "analises_evolucao"):
    """
    Analisa a evolução individual de um aluno
    """
    # Extrair dados de evolução
    evolucao = []
    for i, dia_dados in enumerate(dados_aluno):
        evolucao.append({
            'dia': i + 1,
            'ndcg5': dia_dados['ndcg5']
        })
    
    df_evolucao = pd.DataFrame(evolucao)
    
    # Calcular estatísticas
    primeiro_dia = df_evolucao.iloc[0]['ndcg5']
    quinto_dia = df_evolucao.iloc[4]['ndcg5'] if len(df_evolucao) >= 5 else df_evolucao.iloc[-1]['ndcg5']
    ultimo_dia = df_evolucao.iloc[-1]['ndcg5']
    
    melhoria_5_dias = ((quinto_dia - primeiro_dia) / primeiro_dia) * 100
    melhoria_total = ((ultimo_dia - primeiro_dia) / primeiro_dia) * 100
    
    # Criar gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Evolução completa
    ax1.plot(df_evolucao['dia'], df_evolucao['ndcg5'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_xlabel('Dia')
    ax1.set_ylabel('NDCG@5')
    ax1.set_title(f'Evolução NDCG@5 - Aluno {student_id}')
    ax1.grid(True, alpha=0.3)
    
    # Destacar primeiros 5 dias
    if len(df_evolucao) >= 5:
        ax1.plot(df_evolucao['dia'][:5], df_evolucao['ndcg5'][:5], 'o-', 
                linewidth=3, markersize=8, color='#A23B72', label='Primeiros 5 dias')
        ax1.legend()
    
    # Gráfico 2: Comparação de fases
    fases = ['Dia 1', 'Dia 5', 'Dia Final']
    valores = [primeiro_dia, quinto_dia, ultimo_dia]
    cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax2.bar(fases, valores, color=cores, alpha=0.7)
    ax2.set_ylabel('NDCG@5')
    ax2.set_title(f'Comparação de Fases - Aluno {student_id}')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores):
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
    
    # Retornar estatísticas
    return {
        'student_id': student_id,
        'primeiro_dia': primeiro_dia,
        'quinto_dia': quinto_dia,
        'ultimo_dia': ultimo_dia,
        'melhoria_5_dias': melhoria_5_dias,
        'melhoria_total': melhoria_total,
        'total_dias': len(df_evolucao)
    }

def analisar_evolucao_geral(resultados: Dict, pasta_saida: str = "analises_evolucao"):
    """
    Analisa a evolução geral de todos os alunos
    """
    print("📊 Analisando evolução geral...")
    
    estatisticas = []
    
    for student_id, dados_aluno in resultados.items():
        stats = analisar_evolucao_individual(dados_aluno, student_id, pasta_saida)
        estatisticas.append(stats)
    
    df_stats = pd.DataFrame(estatisticas)
    
    # Criar gráficos gerais
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Distribuição de melhorias nos primeiros 5 dias
    ax1.hist(df_stats['melhoria_5_dias'], bins=10, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.set_xlabel('Melhoria nos Primeiros 5 Dias (%)')
    ax1.set_ylabel('Número de Alunos')
    ax1.set_title('Distribuição de Melhorias (Dias 1-5)')
    ax1.axvline(df_stats['melhoria_5_dias'].mean(), color='red', linestyle='--', 
                label=f'Média: {df_stats["melhoria_5_dias"].mean():.1f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparação NDCG@5 por fase
    fases = ['Dia 1', 'Dia 5', 'Dia Final']
    medias = [df_stats['primeiro_dia'].mean(), df_stats['quinto_dia'].mean(), df_stats['ultimo_dia'].mean()]
    stds = [df_stats['primeiro_dia'].std(), df_stats['quinto_dia'].std(), df_stats['ultimo_dia'].std()]
    
    bars = ax2.bar(fases, medias, yerr=stds, capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    ax2.set_ylabel('NDCG@5 Médio')
    ax2.set_title('NDCG@5 por Fase (Todos os Alunos)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, medias):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Evolução média ao longo dos dias
    # Calcular evolução média para todos os alunos
    max_dias = max(df_stats['total_dias'])
    evolucao_media = []
    
    for dia in range(1, max_dias + 1):
        valores_dia = []
        for student_id, dados_aluno in resultados.items():
            if dia <= len(dados_aluno):
                valores_dia.append(dados_aluno[dia-1]['ndcg5'])
        if valores_dia:
            evolucao_media.append(np.mean(valores_dia))
    
    dias = list(range(1, len(evolucao_media) + 1))
    ax3.plot(dias, evolucao_media, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax3.set_xlabel('Dia')
    ax3.set_ylabel('NDCG@5 Médio')
    ax3.set_title('Evolução Média NDCG@5 (Todos os Alunos)')
    ax3.grid(True, alpha=0.3)
    
    # Destacar primeiros 5 dias
    if len(evolucao_media) >= 5:
        ax3.plot(dias[:5], evolucao_media[:5], 'o-', 
                linewidth=3, markersize=8, color='#A23B72', label='Primeiros 5 dias')
        ax3.legend()
    
    # Gráfico 4: Melhoria vs NDCG@5 final
    ax4.scatter(df_stats['melhoria_5_dias'], df_stats['ultimo_dia'], alpha=0.7, color='#2E86AB')
    ax4.set_xlabel('Melhoria nos Primeiros 5 Dias (%)')
    ax4.set_ylabel('NDCG@5 Final')
    ax4.set_title('Correlação: Melhoria Inicial vs Performance Final')
    ax4.grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df_stats['melhoria_5_dias'], df_stats['ultimo_dia'], 1)
    p = np.poly1d(z)
    ax4.plot(df_stats['melhoria_5_dias'], p(df_stats['melhoria_5_dias']), "r--", alpha=0.8)
    
    plt.tight_layout()
    
    # Salvar gráfico
    nome_arquivo = "evolucao_geral.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_stats

def gerar_relatorio_evolucao(df_stats: pd.DataFrame, pasta_saida: str = "analises_evolucao"):
    """
    Gera relatório detalhado da evolução
    """
    relatorio = f"""
# Relatório de Evolução do NDCG@5

## 📊 Estatísticas Gerais

### Melhorias nos Primeiros 5 Dias:
- **Média**: {df_stats['melhoria_5_dias'].mean():.2f}%
- **Mediana**: {df_stats['melhoria_5_dias'].median():.2f}%
- **Mínimo**: {df_stats['melhoria_5_dias'].min():.2f}%
- **Máximo**: {df_stats['melhoria_5_dias'].max():.2f}%
- **Desvio Padrão**: {df_stats['melhoria_5_dias'].std():.2f}%

### NDCG@5 por Fase:
- **Dia 1 (Média)**: {df_stats['primeiro_dia'].mean():.4f} ± {df_stats['primeiro_dia'].std():.4f}
- **Dia 5 (Média)**: {df_stats['quinto_dia'].mean():.4f} ± {df_stats['quinto_dia'].std():.4f}
- **Dia Final (Média)**: {df_stats['ultimo_dia'].mean():.4f} ± {df_stats['ultimo_dia'].std():.4f}

### Melhoria Total:
- **Média**: {df_stats['melhoria_total'].mean():.2f}%
- **Mediana**: {df_stats['melhoria_total'].median():.2f}%

## 🎯 Análise dos Primeiros 5 Dias

### Alunos com Maior Melhoria (Top 5):
"""
    
    # Top 5 alunos com maior melhoria
    top_melhoria = df_stats.nlargest(5, 'melhoria_5_dias')
    for i, (_, row) in enumerate(top_melhoria.iterrows(), 1):
        relatorio += f"""
{i}. **Aluno {row['student_id']}**:
   - Melhoria: {row['melhoria_5_dias']:.2f}%
   - Dia 1: {row['primeiro_dia']:.4f}
   - Dia 5: {row['quinto_dia']:.4f}
   - Dia Final: {row['ultimo_dia']:.4f}
"""
    
    relatorio += f"""
### Alunos com Menor Melhoria (Bottom 5):
"""
    
    # Bottom 5 alunos com menor melhoria
    bottom_melhoria = df_stats.nsmallest(5, 'melhoria_5_dias')
    for i, (_, row) in enumerate(bottom_melhoria.iterrows(), 1):
        relatorio += f"""
{i}. **Aluno {row['student_id']}**:
   - Melhoria: {row['melhoria_5_dias']:.2f}%
   - Dia 1: {row['primeiro_dia']:.4f}
   - Dia 5: {row['quinto_dia']:.4f}
   - Dia Final: {row['ultimo_dia']:.4f}
"""
    
    relatorio += f"""
## 📈 Insights Principais

1. **Evolução Rápida**: {df_stats['melhoria_5_dias'].mean():.1f}% de melhoria média nos primeiros 5 dias
2. **Estabilidade**: {df_stats['melhoria_5_dias'].std():.1f}% de desvio padrão indica consistência
3. **Impacto Inicial**: {(df_stats['melhoria_5_dias'] / df_stats['melhoria_total'] * 100).mean():.1f}% da melhoria total acontece nos primeiros 5 dias
4. **Performance Final**: NDCG@5 médio final de {df_stats['ultimo_dia'].mean():.4f}

## 🎯 Recomendações

- **Foco nos Primeiros Dias**: A maior parte da melhoria acontece rapidamente
- **Monitoramento Contínuo**: Manter acompanhamento após os primeiros 5 dias
- **Personalização**: Considerar diferentes estratégias baseadas na evolução inicial
"""
    
    # Salvar relatório
    nome_arquivo = "relatorio_evolucao.md"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    print(f"📄 Relatório salvo: {caminho}")

def main():
    parser = argparse.ArgumentParser(description='Analisar evolução do NDCG@5 ao longo dos dias')
    parser.add_argument('--pasta-resultados', default='resultados_ranker_teste', 
                       help='Pasta com resultados do ranker')
    parser.add_argument('--pasta-saida', default='analises_evolucao', 
                       help='Pasta para salvar análises')
    parser.add_argument('--alunos', nargs='+', type=int, 
                       help='IDs específicos dos alunos para analisar (opcional)')
    
    args = parser.parse_args()
    
    print("📊 ANÁLISE DE EVOLUÇÃO DO NDCG@5")
    print("=" * 50)
    
    # Carregar resultados
    resultados = carregar_resultados_ranker(args.pasta_resultados)
    
    if not resultados:
        print("❌ Nenhum resultado encontrado!")
        return
    
    # Filtrar alunos específicos se solicitado
    if args.alunos:
        resultados = {k: v for k, v in resultados.items() if k in args.alunos}
        print(f"📊 Analisando {len(resultados)} alunos específicos...")
    
    # Analisar evolução geral
    df_stats = analisar_evolucao_geral(resultados, args.pasta_saida)
    
    # Gerar relatório
    gerar_relatorio_evolucao(df_stats, args.pasta_saida)
    
    print(f"\n🎉 Análise concluída!")
    print(f"📁 Resultados salvos em: {args.pasta_saida}")
    print(f"📊 Alunos analisados: {len(resultados)}")
    print(f"📈 Melhoria média nos primeiros 5 dias: {df_stats['melhoria_5_dias'].mean():.1f}%")

if __name__ == "__main__":
    main() 