import pandas as pd
import numpy as np
import json
import os
import random
from typing import List, Dict
import argparse

def carregar_base_conteudos(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega a base de conteúdos do ENEM
    """
    df = pd.read_csv(caminho_arquivo)
    
    pesos_classe = {
        'Matemática': 1,
        'Física': 1,
        'Química': 1,
        'Biologia': 1,
        'História': 1,
        'Geografia': 1,
        'Português': 1,
        'Inglês': 1,
        'Espanhol': 1,
        'Arte': 1,
        'Educação Física': 1,
        'Filosofia': 1,
        'Sociologia': 1
    }
    
    df['peso_classe'] = df['classe'].map(pesos_classe)
    
    return df

def gerar_desempenho_aluno(df_base: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """
    Gera dados de desempenho para um aluno baseado na base de conteúdos
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    df_aluno = df_base.copy()
    
    desempenhos = np.random.normal(0, 0.5, len(df_aluno))
    
    desempenhos = np.clip(desempenhos, -1, 1)
    
    df_aluno['desempenho'] = desempenhos
    
    df_aluno['peso_classe'] = df_aluno['peso_classe'].astype(float)
    df_aluno['peso_subclasse'] = df_aluno['peso_subclasse'].astype(float)
    df_aluno['peso_por_questao'] = df_aluno['peso_por_questao'].astype(float)
    
    return df_aluno

def salvar_aluno_csv(df_aluno: pd.DataFrame, student_id: int, pasta_saida: str = "alunos_gerados"):
    """
    Salva os dados do aluno em arquivo CSV
    """
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    nome_arquivo = f"aluno_{student_id}.csv"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    
    df_aluno.to_csv(caminho_arquivo, index=False)
    print(f"✅ Aluno {student_id} salvo em: {caminho_arquivo}")
    
    return caminho_arquivo

def salvar_aluno_json(df_aluno: pd.DataFrame, student_id: int, pasta_saida: str = "alunos_gerados"):
    """
    Salva os dados do aluno em arquivo JSON
    """
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    nome_arquivo = f"aluno_{student_id}.json"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    
    # Converter para formato JSON
    dados_json = []
    for _, row in df_aluno.iterrows():
        dados_json.append({
            "student_id": str(student_id),
            "classe": row["classe"],
            "subclasse": row["subclasse"],
            "desempenho": float(row["desempenho"]),
            "peso_classe": float(row["peso_classe"]),
            "peso_subclasse": float(row["peso_subclasse"]),
            "peso_por_questao": float(row["peso_por_questao"])
        })
    
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Aluno {student_id} (JSON) salvo em: {caminho_arquivo}")
    
    return caminho_arquivo

def gerar_alunos(num_alunos: int, arquivo_base: str = "Alita Start - enem_conteudos_2009_2024_SAS-CNN.csv", 
                 pasta_saida: str = "alunos_gerados", formato: str = "ambos"):
    """
    Gera múltiplos alunos com dados de desempenho
    """
    print(f"🎓 Gerando {num_alunos} alunos...")
    
    print("📚 Carregando base de conteúdos...")
    df_base = carregar_base_conteudos(arquivo_base)
    print(f"   Base carregada com {len(df_base)} conteúdos")
    
    arquivos_gerados = []
    
    for i in range(num_alunos):
        student_id = 1000 + i  # IDs começando em 1000
        
        print(f"\n👤 Gerando aluno {i+1}/{num_alunos} (ID: {student_id})...")
        
        # Gerar dados do aluno
        df_aluno = gerar_desempenho_aluno(df_base, seed=student_id)
        
        # Salvar arquivos
        if formato in ["csv", "ambos"]:
            arquivo_csv = salvar_aluno_csv(df_aluno, student_id, pasta_saida)
            arquivos_gerados.append(arquivo_csv)
        
        if formato in ["json", "ambos"]:
            arquivo_json = salvar_aluno_json(df_aluno, student_id, pasta_saida)
            arquivos_gerados.append(arquivo_json)
    
    print(f"\n🎉 Geração concluída!")
    print(f"   📁 Pasta de saída: {pasta_saida}")
    print(f"   📊 Total de arquivos gerados: {len(arquivos_gerados)}")
    
    return arquivos_gerados

def mostrar_estatisticas_alunos(pasta_alunos: str = "alunos_gerados"):
    """
    Mostra estatísticas dos alunos gerados
    """
    if not os.path.exists(pasta_alunos):
        print("❌ Pasta de alunos não encontrada!")
        return
    
    arquivos_csv = [f for f in os.listdir(pasta_alunos) if f.endswith('.csv')]
    
    if not arquivos_csv:
        print("❌ Nenhum arquivo CSV encontrado!")
        return
    
    print(f"\n📊 Estatísticas dos alunos gerados:")
    print(f"   📁 Pasta: {pasta_alunos}")
    print(f"   👥 Total de alunos: {len(arquivos_csv)}")
    
    for i, arquivo in enumerate(arquivos_csv[:3]):
        caminho = os.path.join(pasta_alunos, arquivo)
        df = pd.read_csv(caminho)
        
        print(f"\n   👤 {arquivo}:")
        print(f"      📚 Conteúdos: {len(df)}")
        print(f"      📈 Desempenho médio: {df['desempenho'].mean():.3f}")
        print(f"      📉 Desempenho mínimo: {df['desempenho'].min():.3f}")
        print(f"      📊 Desempenho máximo: {df['desempenho'].max():.3f}")
        print(f"      🎯 Classes únicas: {df['classe'].nunique()}")

def main():
    parser = argparse.ArgumentParser(description='Gerar múltiplos alunos com dados de desempenho')
    parser.add_argument('num_alunos', type=int, help='Número de alunos a gerar')
    parser.add_argument('--arquivo-base', default='Alita Start - enem_conteudos_2009_2024_SAS-CNN.csv', 
                       help='Arquivo base de conteúdos (padrão: Alita Start - enem_conteudos_2009_2024_SAS-CNN.csv)')
    parser.add_argument('--pasta-saida', default='alunos_gerados', 
                       help='Pasta para salvar os arquivos (padrão: alunos_gerados)')
    parser.add_argument('--formato', choices=['csv', 'json', 'ambos'], default='ambos',
                       help='Formato dos arquivos de saída (padrão: ambos)')
    parser.add_argument('--estatisticas', action='store_true',
                       help='Mostrar estatísticas dos alunos gerados')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.arquivo_base):
        print(f"❌ Arquivo base não encontrado: {args.arquivo_base}")
        return
    
    arquivos_gerados = gerar_alunos(
        num_alunos=args.num_alunos,
        arquivo_base=args.arquivo_base,
        pasta_saida=args.pasta_saida,
        formato=args.formato
    )
    
    if args.estatisticas:
        mostrar_estatisticas_alunos(args.pasta_saida)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("🎓 Gerador de Alunos - ENEM")
        print("=" * 40)
        num_alunos = input("Quantos alunos você quer gerar? ")
        try:
            num_alunos = int(num_alunos)
            gerar_alunos(num_alunos)
            # mostrar_estatisticas_alunos()
        except ValueError:
            print("❌ Por favor, digite um número válido!")
    else:
        main() 