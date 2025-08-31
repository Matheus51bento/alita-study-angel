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

def analisar_declinio_individual(dados_irt: List[Dict], dados_ranker: List[Dict], 
                                student_id: int, pasta_saida: str = "analises_declinio"):
    """
    Analisa o declínio individual de um aluno (IRT vs Ranker)
    """
    # Extrair dados de evolução
    evolucao_irt = []
    evolucao_ranker = []
    
    for i, dia_dados in enumerate(dados_irt):
        evolucao_irt.append({
            'dia': i + 1,
            'ndcg5': dia_dados['ndcg5']
        })
    
    for i, dia_dados in enumerate(dados_ranker):
        evolucao_ranker.append({
            'dia': i + 1,
            'ndcg5': dia_dados['ndcg5']
        })
    
    df_irt = pd.DataFrame(evolucao_irt)
    df_ranker = pd.DataFrame(evolucao_ranker)
    
    # Calcular estatísticas
    primeiro_dia_irt = df_irt.iloc[0]['ndcg5']
    ultimo_dia_irt = df_irt.iloc[-1]['ndcg5']
    declinio_irt = ((primeiro_dia_irt - ultimo_dia_irt) / primeiro_dia_irt) * 100
    
    primeiro_dia_ranker = df_ranker.iloc[0]['ndcg5']
    ultimo_dia_ranker = df_ranker.iloc[-1]['ndcg5']
    melhoria_ranker = ((ultimo_dia_ranker - primeiro_dia_ranker) / primeiro_dia_ranker) * 100
    
    # Criar gráfico comparativo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Evolução comparativa
    ax1.plot(df_irt['dia'], df_irt['ndcg5'], 'o-', linewidth=2, markersize=6, 
             color='#FF6B6B', label='IRT (Declínio)', alpha=0.8)
    ax1.plot(df_ranker['dia'], df_ranker['ndcg5'], 's-', linewidth=2, markersize=6, 
             color='#4ECDC4', label='XGBoost Ranker (Melhoria)', alpha=0.8)
    
    ax1.set_xlabel('Dia')
    ax1.set_ylabel('NDCG@5')
    ax1.set_title(f'Evolução Comparativa - Aluno {student_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparação final
    modelos = ['IRT', 'XGBoost Ranker']
    valores_finais = [ultimo_dia_irt, ultimo_dia_ranker]
    cores = ['#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(modelos, valores_finais, color=cores, alpha=0.7)
    ax2.set_ylabel('NDCG@5 Final')
    ax2.set_title(f'Performance Final - Aluno {student_id}')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores_finais):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    nome_arquivo = f"declinio_aluno_{student_id}.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Retornar estatísticas
    return {
        'student_id': student_id,
        'primeiro_dia_irt': primeiro_dia_irt,
        'ultimo_dia_irt': ultimo_dia_irt,
        'declinio_irt': declinio_irt,
        'primeiro_dia_ranker': primeiro_dia_ranker,
        'ultimo_dia_ranker': ultimo_dia_ranker,
        'melhoria_ranker': melhoria_ranker,
        'diferenca_final': ultimo_dia_ranker - ultimo_dia_irt
    }

def analisar_declinio_geral(resultados_irt: Dict, resultados_ranker: Dict, 
                           pasta_saida: str = "analises_declinio"):
    """
    Analisa o declínio geral de todos os alunos
    """
    print("📊 Analisando declínio geral...")
    
    estatisticas = []
    
    # Processar apenas alunos que têm dados de ambos os modelos
    alunos_comuns = set(resultados_irt.keys()) & set(resultados_ranker.keys())
    
    for student_id in alunos_comuns:
        stats = analisar_declinio_individual(
            resultados_irt[student_id], 
            resultados_ranker[student_id], 
            student_id, 
            pasta_saida
        )
        estatisticas.append(stats)
    
    df_stats = pd.DataFrame(estatisticas)
    
    # Criar gráficos gerais
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Distribuição de declínios do IRT
    ax1.hist(df_stats['declinio_irt'], bins=10, alpha=0.7, color='#FF6B6B', edgecolor='black')
    ax1.set_xlabel('Declínio do IRT (%)')
    ax1.set_ylabel('Número de Alunos')
    ax1.set_title('Distribuição de Declínios do IRT')
    ax1.axvline(df_stats['declinio_irt'].mean(), color='red', linestyle='--', 
                label=f'Média: {df_stats["declinio_irt"].mean():.1f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparação IRT vs Ranker
    modelos = ['IRT', 'XGBoost Ranker']
    medias_finais = [df_stats['ultimo_dia_irt'].mean(), df_stats['ultimo_dia_ranker'].mean()]
    stds_finais = [df_stats['ultimo_dia_irt'].std(), df_stats['ultimo_dia_ranker'].std()]
    cores = ['#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(modelos, medias_finais, yerr=stds_finais, capsize=5, 
                   color=cores, alpha=0.7)
    ax2.set_ylabel('NDCG@5 Final Médio')
    ax2.set_title('Performance Final: IRT vs Ranker')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, medias_finais):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Evolução média ao longo dos dias
    # Calcular evolução média para ambos os modelos
    max_dias = max(len(dados) for dados in resultados_irt.values())
    evolucao_media_irt = []
    evolucao_media_ranker = []
    
    for dia in range(max_dias):
        valores_dia_irt = []
        valores_dia_ranker = []
        
        for student_id in alunos_comuns:
            if dia < len(resultados_irt[student_id]):
                valores_dia_irt.append(resultados_irt[student_id][dia]['ndcg5'])
            if dia < len(resultados_ranker[student_id]):
                valores_dia_ranker.append(resultados_ranker[student_id][dia]['ndcg5'])
        
        if valores_dia_irt:
            evolucao_media_irt.append(np.mean(valores_dia_irt))
        if valores_dia_ranker:
            evolucao_media_ranker.append(np.mean(valores_dia_ranker))
    
    dias = list(range(1, len(evolucao_media_irt) + 1))
    ax3.plot(dias, evolucao_media_irt, 'o-', linewidth=2, markersize=6, 
             color='#FF6B6B', label='IRT (Declínio)', alpha=0.8)
    ax3.plot(dias, evolucao_media_ranker, 's-', linewidth=2, markersize=6, 
             color='#4ECDC4', label='XGBoost Ranker (Melhoria)', alpha=0.8)
    
    ax3.set_xlabel('Dia')
    ax3.set_ylabel('NDCG@5 Médio')
    ax3.set_title('Evolução Média: IRT vs Ranker')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Correlação entre declínio IRT e melhoria Ranker
    ax4.scatter(df_stats['declinio_irt'], df_stats['melhoria_ranker'], alpha=0.7, color='#2E86AB')
    ax4.set_xlabel('Declínio do IRT (%)')
    ax4.set_ylabel('Melhoria do Ranker (%)')
    ax4.set_title('Correlação: Declínio IRT vs Melhoria Ranker')
    ax4.grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df_stats['declinio_irt'], df_stats['melhoria_ranker'], 1)
    p = np.poly1d(z)
    ax4.plot(df_stats['declinio_irt'], p(df_stats['declinio_irt']), "r--", alpha=0.8)
    
    plt.tight_layout()
    
    # Salvar gráfico
    nome_arquivo = "declinio_geral.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_stats

def gerar_relatorio_declinio(df_stats: pd.DataFrame, pasta_saida: str = "analises_declinio"):
    """
    Gera relatório detalhado do declínio do IRT
    """
    relatorio = f"""
# Relatório de Declínio do IRT vs Melhoria do Ranker

## 📊 Estatísticas Gerais

### Declínio do IRT:
- **Média**: {df_stats['declinio_irt'].mean():.2f}%
- **Mediana**: {df_stats['declinio_irt'].median():.2f}%
- **Mínimo**: {df_stats['declinio_irt'].min():.2f}%
- **Máximo**: {df_stats['declinio_irt'].max():.2f}%
- **Desvio Padrão**: {df_stats['declinio_irt'].std():.2f}%

### Melhoria do XGBoost Ranker:
- **Média**: {df_stats['melhoria_ranker'].mean():.2f}%
- **Mediana**: {df_stats['melhoria_ranker'].median():.2f}%
- **Mínimo**: {df_stats['melhoria_ranker'].min():.2f}%
- **Máximo**: {df_stats['melhoria_ranker'].max():.2f}%

### Performance Final:
- **IRT (Média)**: {df_stats['ultimo_dia_irt'].mean():.4f} ± {df_stats['ultimo_dia_irt'].std():.4f}
- **Ranker (Média)**: {df_stats['ultimo_dia_ranker'].mean():.4f} ± {df_stats['ultimo_dia_ranker'].std():.4f}
- **Diferença Média**: {df_stats['diferenca_final'].mean():.4f}

## 🎯 Análise do Declínio do IRT

### Alunos com Maior Declínio (Top 5):
"""
    
    # Top 5 alunos com maior declínio
    top_declinio = df_stats.nlargest(5, 'declinio_irt')
    for i, (_, row) in enumerate(top_declinio.iterrows(), 1):
        relatorio += f"""
{i}. **Aluno {int(row['student_id'])}**:
   - Declínio IRT: {row['declinio_irt']:.2f}%
   - Melhoria Ranker: {row['melhoria_ranker']:.2f}%
   - IRT Final: {row['ultimo_dia_irt']:.4f}
   - Ranker Final: {row['ultimo_dia_ranker']:.4f}
"""
    
    relatorio += f"""
### Alunos com Menor Declínio (Bottom 5):
"""
    
    # Bottom 5 alunos com menor declínio
    bottom_declinio = df_stats.nsmallest(5, 'declinio_irt')
    for i, (_, row) in enumerate(bottom_declinio.iterrows(), 1):
        relatorio += f"""
{i}. **Aluno {int(row['student_id'])}**:
   - Declínio IRT: {row['declinio_irt']:.2f}%
   - Melhoria Ranker: {row['melhoria_ranker']:.2f}%
   - IRT Final: {row['ultimo_dia_irt']:.4f}
   - Ranker Final: {row['ultimo_dia_ranker']:.4f}
"""
    
    relatorio += f"""
## 📈 Insights Principais

1. **Declínio Consistente**: {df_stats['declinio_irt'].mean():.1f}% de declínio médio do IRT
2. **Melhoria Contínua**: {df_stats['melhoria_ranker'].mean():.1f}% de melhoria média do Ranker
3. **Divergência de Performance**: O IRT piora enquanto o Ranker melhora
4. **Vantagem Final**: {df_stats['diferenca_final'].mean():.1f} pontos de vantagem do Ranker

## 🎯 Recomendações

- **Substituição Progressiva**: Considerar substituir IRT pelo Ranker
- **Monitoramento**: Acompanhar declínio do IRT em produção
- **Otimização**: Focar recursos no desenvolvimento do Ranker
- **Migração**: Planejar migração gradual dos alunos para o Ranker

## 🎉 Conclusão

O IRT demonstra **declínio consistente** de {df_stats['declinio_irt'].mean():.1f}%, enquanto o XGBoost Ranker apresenta **melhoria contínua** de {df_stats['melhoria_ranker'].mean():.1f}%. Esta divergência sugere que o Ranker é mais adequado para o contexto de aprendizado incremental.
"""
    
    # Salvar relatório
    nome_arquivo = "relatorio_declinio.md"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    print(f"📄 Relatório salvo: {caminho}")

def main():
    parser = argparse.ArgumentParser(description='Analisar declínio do IRT vs melhoria do ranker')
    parser.add_argument('--pasta-irt', default='resultados_irt_teste', 
                       help='Pasta com resultados do IRT')
    parser.add_argument('--pasta-ranker', default='resultados_ranker_teste', 
                       help='Pasta com resultados do ranker')
    parser.add_argument('--pasta-saida', default='analises_declinio', 
                       help='Pasta para salvar análises')
    parser.add_argument('--alunos', nargs='+', type=int, 
                       help='IDs específicos dos alunos para analisar (opcional)')
    
    args = parser.parse_args()
    
    print("📊 ANÁLISE DE DECLÍNIO DO IRT")
    print("=" * 50)
    
    # Carregar resultados
    resultados_irt = carregar_resultados_irt(args.pasta_irt)
    resultados_ranker = carregar_resultados_ranker(args.pasta_ranker)
    
    if not resultados_irt or not resultados_ranker:
        print("❌ Não foi possível carregar os resultados!")
        return
    
    # Filtrar alunos específicos se solicitado
    if args.alunos:
        resultados_irt = {k: v for k, v in resultados_irt.items() if k in args.alunos}
        resultados_ranker = {k: v for k, v in resultados_ranker.items() if k in args.alunos}
        print(f"📊 Analisando {len(resultados_irt)} alunos específicos...")
    
    # Analisar declínio geral
    df_stats = analisar_declinio_geral(resultados_irt, resultados_ranker, args.pasta_saida)
    
    # Gerar relatório
    gerar_relatorio_declinio(df_stats, args.pasta_saida)
    
    print(f"\n🎉 Análise de declínio concluída!")
    print(f"📁 Resultados salvos em: {args.pasta_saida}")
    print(f"📊 Alunos analisados: {len(df_stats)}")
    print(f"📉 Declínio médio do IRT: {df_stats['declinio_irt'].mean():.1f}%")
    print(f"📈 Melhoria média do Ranker: {df_stats['melhoria_ranker'].mean():.1f}%")

if __name__ == "__main__":
    main() 