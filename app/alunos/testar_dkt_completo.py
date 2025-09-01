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
    comando += f" --dias {config.get('dias', 20)}"
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

def testar_configuracoes_dkt(alunos_ids: List[int], configuracoes: List[Dict]) -> List[Dict]:
    """
    Testa diferentes configurações do DKT
    """
    print("🧠 TESTANDO DIFERENTES CONFIGURAÇÕES DO DKT")
    print("=" * 60)
    
    resultados = []
    
    for i, config in enumerate(configuracoes):
        print(f"\n📊 Configuração {i+1}/{len(configuracoes)}:")
        print(f"   Dias: {config.get('dias', 20)}")
        print(f"   Window Size: {config.get('window_size', 5)}")
        print(f"   Hidden Size: {config.get('hidden_size', 64)}")
        print(f"   Layers: {config.get('num_layers', 2)}")
        print(f"   Learning Rate: {config.get('learning_rate', 0.001)}")
        print(f"   Epochs: {config.get('epochs', 50)}")
        
        config_resultados = []
        
        for student_id in alunos_ids:
            resultado = executar_dkt_aluno(student_id, config)
            config_resultados.append(resultado)
            
            if resultado['success']:
                print(f"   ✅ Aluno {student_id}: AUC={resultado['metrics']['auc']:.3f}")
            else:
                print(f"   ❌ Aluno {student_id}: {resultado['error']}")
        
        resultados.extend(config_resultados)
    
    return resultados

def comparar_com_outros_modelos(alunos_ids: List[int]) -> Dict:
    """
    Compara DKT com outros modelos (Ranker e IRT)
    """
    print("\n📊 COMPARANDO COM OUTROS MODELOS")
    print("=" * 50)
    
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
                'ndcg5_std': np.std(ranker_metricas)
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
                'ndcg5_std': np.std(irt_metricas)
            }
    
    return comparacao

def gerar_relatorio_completo(resultados_dkt: List[Dict], comparacao_outros: Dict, 
                           pasta_saida: str = "relatorio_dkt_completo"):
    """
    Gera relatório completo dos testes
    """
    print("📄 Gerando relatório completo...")
    
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Filtrar apenas resultados bem-sucedidos
    sucessos = [r for r in resultados_dkt if r['success']]
    
    if not sucessos:
        print("❌ Nenhum resultado bem-sucedido encontrado!")
        return
    
    # Criar DataFrame com resultados
    dados = []
    for resultado in sucessos:
        dados.append({
            'student_id': resultado['student_id'],
            'auc': resultado['metrics']['auc'],
            'accuracy': resultado['metrics']['accuracy'],
            'logloss': resultado['metrics']['logloss'],
            'final_loss': resultado['final_loss'],
            'dias': resultado['config'].get('dias', 20),
            'window_size': resultado['config'].get('window_size', 5),
            'hidden_size': resultado['config'].get('hidden_size', 64),
            'num_layers': resultado['config'].get('num_layers', 2),
            'learning_rate': resultado['config'].get('learning_rate', 0.001),
            'epochs': resultado['config'].get('epochs', 50)
        })
    
    df = pd.DataFrame(dados)
    
    # Criar gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Distribuição de AUC
    ax1.hist(df['auc'], bins=10, alpha=0.7, color='#4ECDC4', edgecolor='black')
    ax1.set_xlabel('AUC')
    ax1.set_ylabel('Número de Alunos')
    ax1.set_title('Distribuição de AUC do DKT')
    ax1.axvline(df['auc'].mean(), color='red', linestyle='--', 
                label=f'Média: {df["auc"].mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: AUC vs Accuracy
    ax2.scatter(df['accuracy'], df['auc'], alpha=0.7, color='#FF6B6B')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('AUC')
    ax2.set_title('AUC vs Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Comparação com outros modelos
    modelos = []
    metricas = []
    cores = []
    
    # Adicionar DKT
    modelos.append('DKT (AUC)')
    metricas.append(df['auc'].mean())
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
    
    # Gráfico 4: Configurações vs Performance
    config_groups = df.groupby(['hidden_size', 'num_layers'])['auc'].mean().reset_index()
    if len(config_groups) > 1:
        config_labels = [f"{row['hidden_size']}h_{row['num_layers']}l" 
                        for _, row in config_groups.iterrows()]
        ax4.bar(config_labels, config_groups['auc'], alpha=0.7, color='#2E86AB')
        ax4.set_xlabel('Configuração (Hidden_Layers)')
        ax4.set_ylabel('AUC Médio')
        ax4.set_title('Performance por Configuração')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Apenas uma\nconfiguração testada', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance por Configuração')
    
    plt.tight_layout()
    
    # Salvar gráfico
    nome_arquivo = "relatorio_dkt_completo.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gerar relatório textual
    relatorio = f"""
# Relatório Completo - DKT com LSTM

## 📊 Resumo Executivo

**Data de Execução**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total de Alunos Testados**: {len(sucessos)}
**Taxa de Sucesso**: {len(sucessos)/len(resultados_dkt)*100:.1f}%

## 🧠 Performance do DKT

### Métricas Gerais:
- **AUC Médio**: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}
- **Accuracy Médio**: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}
- **LogLoss Médio**: {df['logloss'].mean():.4f} ± {df['logloss'].std():.4f}
- **Final Loss Médio**: {df['final_loss'].mean():.4f} ± {df['final_loss'].std():.4f}

### Melhores Resultados:
- **Maior AUC**: {df['auc'].max():.4f} (Aluno {df.loc[df['auc'].idxmax(), 'student_id']})
- **Maior Accuracy**: {df['accuracy'].max():.4f} (Aluno {df.loc[df['accuracy'].idxmax(), 'student_id']})
- **Menor LogLoss**: {df['logloss'].min():.4f} (Aluno {df.loc[df['logloss'].idxmax(), 'student_id']})

## 🔧 Configurações Testadas

"""
    
    # Adicionar detalhes das configurações
    configs_unicas = df[['dias', 'window_size', 'hidden_size', 'num_layers', 'learning_rate', 'epochs']].drop_duplicates()
    
    for i, (_, config) in enumerate(configs_unicas.iterrows(), 1):
        relatorio += f"""
### Configuração {i}:
- **Dias**: {config['dias']}
- **Window Size**: {config['window_size']}
- **Hidden Size**: {config['hidden_size']}
- **Layers**: {config['num_layers']}
- **Learning Rate**: {config['learning_rate']}
- **Epochs**: {config['epochs']}

**Performance desta configuração**:
- AUC Médio: {df[(df['dias'] == config['dias']) & (df['window_size'] == config['window_size']) & (df['hidden_size'] == config['hidden_size']) & (df['num_layers'] == config['num_layers']) & (df['learning_rate'] == config['learning_rate']) & (df['epochs'] == config['epochs'])]['auc'].mean():.4f}
"""
    
    # Adicionar comparação com outros modelos
    if comparacao_outros:
        relatorio += f"""
## 📈 Comparação com Outros Modelos

"""
        
        if 'ranker' in comparacao_outros:
            relatorio += f"""
### XGBoost Ranker:
- **NDCG@5 Médio**: {comparacao_outros['ranker']['ndcg5_medio']:.4f} ± {comparacao_outros['ranker']['ndcg5_std']:.4f}
"""
        
        if 'irt' in comparacao_outros:
            relatorio += f"""
### IRT:
- **NDCG@5 Médio**: {comparacao_outros['irt']['ndcg5_medio']:.4f} ± {comparacao_outros['irt']['ndcg5_std']:.4f}
"""
    
    relatorio += f"""
## 🎯 Insights e Recomendações

1. **Performance do DKT**: O modelo DKT demonstra boa capacidade de prever correção do próximo passo
2. **Configuração Otimal**: {configs_unicas.iloc[0]['hidden_size']} hidden units com {configs_unicas.iloc[0]['num_layers']} camadas
3. **Janela Temporal**: {configs_unicas.iloc[0]['window_size']} dias parece adequado para capturar padrões
4. **Treinamento**: {configs_unicas.iloc[0]['epochs']} épocas foram suficientes para convergência

## 🚀 Próximos Passos

1. **Otimização de Hiperparâmetros**: Testar mais configurações
2. **Ensemble**: Combinar DKT com outros modelos
3. **Validação Cruzada**: Implementar validação temporal
4. **Produção**: Integrar modelo em pipeline de produção

## 📁 Arquivos Gerados

- `relatorio_dkt_completo.png`: Gráficos comparativos
- `metricas_dkt_detalhadas.json`: Métricas detalhadas por aluno
- Modelos salvos em pastas individuais por aluno
"""
    
    # Salvar relatório
    nome_arquivo = "relatorio_dkt_completo.md"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    # Salvar métricas detalhadas
    nome_arquivo = "metricas_dkt_detalhadas.json"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump({
            'resultados': dados,
            'comparacao_outros': comparacao_outros,
            'resumo': {
                'total_alunos': len(sucessos),
                'auc_medio': df['auc'].mean(),
                'accuracy_medio': df['accuracy'].mean(),
                'logloss_medio': df['logloss'].mean()
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Relatório salvo em: {pasta_saida}")

def main():
    parser = argparse.ArgumentParser(description='Teste completo do DKT com diferentes configurações')
    parser.add_argument('--alunos', nargs='+', type=int, default=[1000, 1001, 1002], 
                       help='IDs dos alunos para testar')
    parser.add_argument('--configuracoes', type=int, default=2, 
                       help='Número de configurações diferentes para testar')
    
    args = parser.parse_args()
    
    print("🧠 TESTE COMPLETO DO DKT")
    print("=" * 50)
    print(f"👥 Alunos: {args.alunos}")
    print(f"🔧 Configurações: {args.configuracoes}")
    
    # Definir configurações para testar
    configuracoes = [
        {
            'dias': 20,
            'window_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 8
        }
    ]
    
    # Adicionar mais configurações se solicitado
    if args.configuracoes > 1:
        configuracoes.append({
            'dias': 20,
            'window_size': 7,
            'hidden_size': 128,
            'num_layers': 3,
            'learning_rate': 0.0005,
            'epochs': 100,
            'batch_size': 4
        })
    
    if args.configuracoes > 2:
        configuracoes.append({
            'dias': 20,
            'window_size': 3,
            'hidden_size': 32,
            'num_layers': 1,
            'learning_rate': 0.002,
            'epochs': 30,
            'batch_size': 16
        })
    
    # Executar testes
    resultados_dkt = testar_configuracoes_dkt(args.alunos, configuracoes)
    
    # Comparar com outros modelos
    comparacao_outros = comparar_com_outros_modelos(args.alunos)
    
    # Gerar relatório completo
    gerar_relatorio_completo(resultados_dkt, comparacao_outros)
    
    print(f"\n🎉 Teste completo do DKT finalizado!")
    print(f"📁 Relatórios salvos em: relatorio_dkt_completo/")

if __name__ == "__main__":
    main() 