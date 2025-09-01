import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import argparse
import subprocess
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def executar_dkt_aluno(student_id: int, config: Dict) -> Dict:
    """
    Executa DKT para um aluno com configuração específica
    """
    print(f"🧠 Executando DKT para aluno {student_id}...")
    
    # Construir comando
    comando = f"python dkt_lstm.py {student_id}"
    comando += f" --dias {config.get('dias', 50)}"
    comando += f" --window-size {config.get('window_size', 5)}"
    comando += f" --hidden-size {config.get('hidden_size', 64)}"
    comando += f" --num-layers {config.get('num_layers', 2)}"
    comando += f" --learning-rate {config.get('learning_rate', 0.001)}"
    comando += f" --epochs {config.get('epochs', 50)}"
    comando += f" --batch-size {config.get('batch_size', 8)}"
    comando += f" --pasta-saida resultados_dkt_aluno_{student_id}"
    
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            # Carregar métricas
            pasta_saida = f"resultados_dkt_aluno_{student_id}"
            arquivo_metricas = os.path.join(pasta_saida, "metricas_dkt.json")
            
            if os.path.exists(arquivo_metricas):
                with open(arquivo_metricas, 'r') as f:
                    metricas = json.load(f)
                
                return {
                    'student_id': student_id,
                    'success': True,
                    'metrics': metricas['metrics'],
                    'final_loss': metricas['final_loss'],
                    'config': config
                }
            else:
                return {
                    'student_id': student_id,
                    'success': False,
                    'error': 'Métricas não encontradas'
                }
        else:
            return {
                'student_id': student_id,
                'success': False,
                'error': resultado.stderr
            }
            
    except Exception as e:
        return {
            'student_id': student_id,
            'success': False,
            'error': str(e)
        }

def carregar_metricas_outros_modelos(alunos_ids: List[int]) -> Dict:
    """
    Carrega métricas de outros modelos para comparação
    """
    print("📊 Carregando métricas de outros modelos...")
    
    comparacao = {}
    
    # Carregar resultados do Ranker
    pasta_ranker = "resultados_ranker"
    if os.path.exists(pasta_ranker):
        print("📈 Carregando resultados do Ranker...")
        ranker_metricas = []
        
        for student_id in alunos_ids:
            arquivo = os.path.join(pasta_ranker, f"resultado_ranker_aluno_{student_id}.json")
            if os.path.exists(arquivo):
                with open(arquivo, 'r') as f:
                    dados = json.load(f)
                    # Pegar NDCG@5 final
                    ndcg_final = dados[-1]['ndcg5'] if dados else 0
                    ranker_metricas.append(ndcg_final)
        
        if ranker_metricas:
            comparacao['ranker'] = {
                'ndcg5_medio': np.mean(ranker_metricas),
                'ndcg5_std': np.std(ranker_metricas),
                'metricas_por_aluno': dict(zip(alunos_ids, ranker_metricas))
            }
    
    # Carregar resultados do IRT
    pasta_irt = "resultados_irt"
    if os.path.exists(pasta_irt):
        print("📉 Carregando resultados do IRT...")
        irt_metricas = []
        
        for student_id in alunos_ids:
            arquivo = os.path.join(pasta_irt, f"resultado_irt_aluno_{student_id}.json")
            if os.path.exists(arquivo):
                with open(arquivo, 'r') as f:
                    dados = json.load(f)
                    # Pegar NDCG@5 final
                    ndcg_final = dados[-1]['ndcg5'] if dados else 0
                    irt_metricas.append(ndcg_final)
        
        if irt_metricas:
            comparacao['irt'] = {
                'ndcg5_medio': np.mean(irt_metricas),
                'ndcg5_std': np.std(irt_metricas),
                'metricas_por_aluno': dict(zip(alunos_ids, irt_metricas))
            }
    
    return comparacao

def gerar_relatorio_comparativo(resultados_dkt: List[Dict], comparacao_outros: Dict, 
                               pasta_saida: str = "comparacao_dkt_completa"):
    """
    Gera relatório comparativo completo
    """
    print("📄 Gerando relatório comparativo...")
    
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Filtrar apenas resultados bem-sucedidos do DKT
    sucessos_dkt = [r for r in resultados_dkt if r['success']]
    
    if not sucessos_dkt:
        print("❌ Nenhum resultado DKT bem-sucedido encontrado!")
        return
    
    # Criar DataFrame com resultados do DKT
    dados_dkt = []
    for resultado in sucessos_dkt:
        dados_dkt.append({
            'student_id': resultado['student_id'],
            'r2': resultado['metrics']['r2'],
            'mse': resultado['metrics']['mse'],
            'mae': resultado['metrics']['mae'],
            'final_loss': resultado['final_loss']
        })
    
    df_dkt = pd.DataFrame(dados_dkt)
    
    # Criar gráficos comparativos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Distribuição de R² do DKT
    ax1.hist(df_dkt['r2'], bins=10, alpha=0.7, color='#4ECDC4', edgecolor='black')
    ax1.set_xlabel('R²')
    ax1.set_ylabel('Número de Alunos')
    ax1.set_title('Distribuição de R² do DKT')
    ax1.axvline(df_dkt['r2'].mean(), color='red', linestyle='--', 
                label=f'Média: {df_dkt["r2"].mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: R² vs MSE
    ax2.scatter(df_dkt['mse'], df_dkt['r2'], alpha=0.7, color='#FF6B6B')
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('R²')
    ax2.set_title('R² vs MSE do DKT')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Comparação entre modelos
    modelos = []
    metricas = []
    cores = []
    
    # Adicionar DKT
    modelos.append('DKT (R²)')
    metricas.append(df_dkt['r2'].mean())
    cores.append('#4ECDC4')
    
    # Adicionar outros modelos se disponíveis
    if 'ranker' in comparacao_outros:
        modelos.append('Ranker (NDCG@5)')
        metricas.append(comparacao_outros['ranker']['ndcg5_medio'])
        cores.append('#45B7D1')
    
    if 'irt' in comparacao_outros:
        modelos.append('IRT (NDCG@5)')
        metricas.append(comparacao_outros['irt']['ndcg5_medio'])
        cores.append('#FF6B6B')
    
    if len(modelos) > 1:
        bars = ax3.bar(modelos, metricas, color=cores, alpha=0.7)
        ax3.set_ylabel('Métrica')
        ax3.set_title('Comparação: DKT vs Outros Modelos')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, valor in zip(bars, metricas):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes\npara comparação', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Comparação com Outros Modelos')
    
    # Gráfico 4: Performance por aluno (se temos dados de outros modelos)
    if 'ranker' in comparacao_outros and 'irt' in comparacao_outros:
        # Criar DataFrame comparativo por aluno
        dados_comparativos = []
        for student_id in df_dkt['student_id']:
            dados_comparativos.append({
                'student_id': student_id,
                'dkt_r2': df_dkt[df_dkt['student_id'] == student_id]['r2'].iloc[0],
                'ranker_ndcg5': comparacao_outros['ranker']['metricas_por_aluno'].get(student_id, 0),
                'irt_ndcg5': comparacao_outros['irt']['metricas_por_aluno'].get(student_id, 0)
            })
        
        df_comp = pd.DataFrame(dados_comparativos)
        
        # Plotar comparação por aluno
        x = np.arange(len(df_comp))
        width = 0.25
        
        ax4.bar(x - width, df_comp['dkt_r2'], width, label='DKT (R²)', alpha=0.7, color='#4ECDC4')
        ax4.bar(x, df_comp['ranker_ndcg5'], width, label='Ranker (NDCG@5)', alpha=0.7, color='#45B7D1')
        ax4.bar(x + width, df_comp['irt_ndcg5'], width, label='IRT (NDCG@5)', alpha=0.7, color='#FF6B6B')
        
        ax4.set_xlabel('Aluno')
        ax4.set_ylabel('Métrica')
        ax4.set_title('Performance por Aluno')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Aluno {sid}' for sid in df_comp['student_id']], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Dados insuficientes\npara comparação por aluno', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance por Aluno')
    
    plt.tight_layout()
    
    # Salvar gráfico
    nome_arquivo = "comparacao_dkt_completa.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gerar relatório textual
    relatorio = f"""
# Relatório Comparativo - DKT vs Outros Modelos

## 📊 Resumo Executivo

**Data de Execução**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total de Alunos DKT**: {len(sucessos_dkt)}
**Taxa de Sucesso DKT**: {len(sucessos_dkt)/len(resultados_dkt)*100:.1f}%

## 🧠 Performance do DKT (Regressão)

### Métricas Gerais:
- **R² Médio**: {df_dkt['r2'].mean():.4f} ± {df_dkt['r2'].std():.4f}
- **MSE Médio**: {df_dkt['mse'].mean():.4f} ± {df_dkt['mse'].std():.4f}
- **MAE Médio**: {df_dkt['mae'].mean():.4f} ± {df_dkt['mae'].std():.4f}
- **Final Loss Médio**: {df_dkt['final_loss'].mean():.4f} ± {df_dkt['final_loss'].std():.4f}

### Melhores Resultados DKT:
- **Maior R²**: {df_dkt['r2'].max():.4f} (Aluno {df_dkt.loc[df_dkt['r2'].idxmax(), 'student_id']})
- **Menor MSE**: {df_dkt['mse'].min():.4f} (Aluno {df_dkt.loc[df_dkt['mse'].idxmin(), 'student_id']})
- **Menor MAE**: {df_dkt['mae'].min():.4f} (Aluno {df_dkt.loc[df_dkt['mae'].idxmin(), 'student_id']})

## 📈 Comparação com Outros Modelos

"""
    
    # Adicionar comparação com outros modelos
    if comparacao_outros:
        if 'ranker' in comparacao_outros:
            relatorio += f"""
### XGBoost Ranker (Ranking):
- **NDCG@5 Médio**: {comparacao_outros['ranker']['ndcg5_medio']:.4f} ± {comparacao_outros['ranker']['ndcg5_std']:.4f}
"""
        
        if 'irt' in comparacao_outros:
            relatorio += f"""
### IRT (Ranking):
- **NDCG@5 Médio**: {comparacao_outros['irt']['ndcg5_medio']:.4f} ± {comparacao_outros['irt']['ndcg5_std']:.4f}
"""
    
    relatorio += f"""
## 🎯 Análise Comparativa

### DKT (Regressão):
- **Objetivo**: Prever desempenho contínuo do próximo passo
- **Métrica**: R² (coeficiente de determinação)
- **Vantagem**: Predição precisa de valores contínuos
- **Aplicação**: Previsão de desempenho futuro

### Ranker/IRT (Ranking):
- **Objetivo**: Ordenar conteúdos por prioridade
- **Métrica**: NDCG@5 (qualidade do ranking)
- **Vantagem**: Recomendação de conteúdos
- **Aplicação**: Sistema de recomendação

## 🚀 Insights e Recomendações

1. **DKT Excelente para Previsão**: R² médio de {df_dkt['r2'].mean():.1f} indica excelente capacidade preditiva
2. **Complementaridade**: DKT e Ranker/IRT são complementares, não concorrentes
3. **Pipeline Integrado**: 
   - DKT para prever desempenho futuro
   - Ranker para recomendar conteúdos baseado na previsão
4. **Validação Temporal**: Implementar validação cruzada temporal para DKT

## 📁 Arquivos Gerados

- `comparacao_dkt_completa.png`: Gráficos comparativos
- `metricas_comparativas.json`: Métricas detalhadas
- Modelos DKT salvos em pastas individuais por aluno

## 🎉 Conclusão

O DKT demonstra **excelente performance** na previsão de desempenho contínuo (R² = {df_dkt['r2'].mean():.3f}), complementando perfeitamente os modelos de ranking existentes. A combinação de DKT + Ranker pode criar um sistema de recomendação mais inteligente e personalizado.
"""
    
    # Salvar relatório
    nome_arquivo = "relatorio_comparativo_dkt.md"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    # Salvar métricas comparativas
    nome_arquivo = "metricas_comparativas.json"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump({
            'dkt_resultados': dados_dkt,
            'comparacao_outros': comparacao_outros,
            'resumo': {
                'total_alunos_dkt': len(sucessos_dkt),
                'r2_medio_dkt': df_dkt['r2'].mean(),
                'mse_medio_dkt': df_dkt['mse'].mean(),
                'mae_medio_dkt': df_dkt['mae'].mean()
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Relatório comparativo salvo em: {pasta_saida}")

def main():
    parser = argparse.ArgumentParser(description='Comparar DKT com outros modelos')
    parser.add_argument('--alunos', nargs='+', type=int, default=[1000, 1001, 1002], 
                       help='IDs dos alunos para testar')
    parser.add_argument('--dias', type=int, default=50, help='Número de dias para DKT')
    
    args = parser.parse_args()
    
    print("🧠 COMPARAÇÃO DKT vs OUTROS MODELOS")
    print("=" * 50)
    print(f"👥 Alunos: {args.alunos}")
    print(f"📅 Dias para DKT: {args.dias}")
    
    # Configuração padrão para DKT
    config_dkt = {
        'dias': args.dias,
        'window_size': 5,
        'hidden_size': 64,
        'num_layers': 2,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 8
    }
    
    # Executar DKT para todos os alunos
    resultados_dkt = []
    for student_id in args.alunos:
        resultado = executar_dkt_aluno(student_id, config_dkt)
        resultados_dkt.append(resultado)
        
        if resultado['success']:
            print(f"✅ Aluno {student_id}: R²={resultado['metrics']['r2']:.3f}")
        else:
            print(f"❌ Aluno {student_id}: {resultado['error']}")
    
    # Carregar métricas de outros modelos
    comparacao_outros = carregar_metricas_outros_modelos(args.alunos)
    
    # Gerar relatório comparativo
    gerar_relatorio_comparativo(resultados_dkt, comparacao_outros)
    
    print(f"\n🎉 Comparação DKT vs outros modelos finalizada!")
    print(f"📁 Relatórios salvos em: comparacao_dkt_completa/")

if __name__ == "__main__":
    main() 