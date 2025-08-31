import os
import subprocess
import sys
import time
import asyncio
import concurrent.futures
from datetime import datetime
import argparse
from typing import List, Tuple
import threading
from queue import Queue

# Variável global para controlar o número de workers ativos
workers_ativos = 0
lock = threading.Lock()

def executar_comando(comando: str, descricao: str) -> bool:
    """
    Executa um comando e mostra o progresso
    """
    global workers_ativos
    
    with lock:
        workers_ativos += 1
        worker_id = workers_ativos
    
    print(f"\n🔄 [{worker_id}] {descricao}...")
    print(f"   [{worker_id}] Comando: {comando}")
    
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"   [{worker_id}] ✅ {descricao} concluído com sucesso!")
            return True
        else:
            print(f"   [{worker_id}] ❌ Erro em {descricao}:")
            print(f"   [{worker_id}] {resultado.stderr}")
            return False
            
    except Exception as e:
        print(f"   [{worker_id}] ❌ Erro ao executar {descricao}: {e}")
        return False
    finally:
        with lock:
            workers_ativos -= 1

def gerar_alunos(num_alunos: int, pasta_saida: str = "alunos_teste") -> bool:
    """
    Gera múltiplos alunos
    """
    comando = f"python gerar_alunos.py {num_alunos} --pasta-saida {pasta_saida}"
    return executar_comando(comando, f"Gerando {num_alunos} alunos")

def simular_desempenho_aluno(student_id: int, dias: int, pasta_alunos: str = "alunos_teste") -> bool:
    """
    Simula desempenho para um aluno específico
    """
    arquivo_csv = f"{pasta_alunos}/aluno_{student_id}.csv"
    comando = f"python gerar_simulacao.py {arquivo_csv} {student_id} --dias {dias}"
    return executar_comando(comando, f"Simulando {dias} dias para aluno {student_id}")

def treinar_ranker_aluno(student_id: int, pasta_saida: str = "resultados_ranker_teste") -> bool:
    """
    Treina ranker para um aluno específico
    """
    pasta_aluno = f"output/aluno_{student_id}"
    comando = f"python treinar_ranker_incremental.py {pasta_aluno} --pasta-saida {pasta_saida}"
    return executar_comando(comando, f"Treinando ranker para aluno {student_id}")

def treinar_irt_aluno(student_id: int, dias: int, pasta_alunos: str = "alunos_teste", pasta_saida: str = "resultados_irt_teste") -> bool:
    """
    Treina IRT para um aluno específico
    """
    arquivo_csv = f"{pasta_alunos}/aluno_{student_id}.csv"
    comando = f"python treinar_irt_incremental.py {arquivo_csv} --dias {dias} --pasta-saida {pasta_saida}"
    return executar_comando(comando, f"Treinando IRT para aluno {student_id}")

def processar_aluno(student_id: int, dias: int, pasta_alunos: str = "alunos_teste") -> Tuple[int, bool]:
    """
    Processa um aluno completo: simulação + treinamento ranker + treinamento IRT
    """
    print(f"\n🎯 Processando aluno {student_id}...")
    
    # 1. Simular desempenho
    if not simular_desempenho_aluno(student_id, dias, pasta_alunos):
        return student_id, False
    
    # 2. Treinar ranker
    if not treinar_ranker_aluno(student_id):
        return student_id, False
    
    # 3. Treinar IRT
    if not treinar_irt_aluno(student_id, dias, pasta_alunos):
        return student_id, False
    
    print(f"✅ Aluno {student_id} processado com sucesso!")
    return student_id, True

def processar_alunos_paralelo(alunos_ids: List[int], dias: int, pasta_alunos: str, max_workers: int = 4) -> Tuple[int, int]:
    """
    Processa múltiplos alunos em paralelo usando ThreadPoolExecutor
    """
    sucessos = 0
    falhas = 0
    
    print(f"\n🚀 Processando {len(alunos_ids)} alunos em paralelo (max {max_workers} workers)")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todas as tarefas
        future_to_student = {
            executor.submit(processar_aluno, student_id, dias, pasta_alunos): student_id 
            for student_id in alunos_ids
        }
        
        # Processar resultados conforme completam
        for i, future in enumerate(concurrent.futures.as_completed(future_to_student)):
            student_id = future_to_student[future]
            
            try:
                student_id_result, sucesso = future.result()
                if sucesso:
                    sucessos += 1
                else:
                    falhas += 1
                    print(f"⚠️  Aluno {student_id} falhou")
                
                # Mostrar progresso
                total_processados = i + 1
                print(f"\n📈 Progresso: {total_processados}/{len(alunos_ids)} ({(total_processados)/len(alunos_ids)*100:.1f}%)")
                print(f"   ✅ Sucessos: {sucessos} | ❌ Falhas: {falhas}")
                
            except Exception as e:
                falhas += 1
                print(f"❌ Erro ao processar aluno {student_id}: {e}")
    
    return sucessos, falhas

def executar_teste_completo_async(num_alunos: int = 50, dias: int = 100, pasta_alunos: str = "alunos_teste", max_workers: int = 4) -> bool:
    """
    Executa o teste completo para todos os alunos de forma assíncrona
    """
    print("🚀 INICIANDO TESTE COMPLETO ASSÍNCRONO")
    print("=" * 60)
    print(f"📊 Configuração:")
    print(f"   👥 Alunos: {num_alunos}")
    print(f"   📅 Dias de simulação: {dias}")
    print(f"   📁 Pasta de alunos: {pasta_alunos}")
    print(f"   🔄 Workers paralelos: {max_workers}")
    print(f"   🕐 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Gerar alunos
    if not gerar_alunos(num_alunos, pasta_alunos):
        print("❌ Falha ao gerar alunos. Abortando teste.")
        return False
    
    # 2. Preparar lista de IDs dos alunos
    alunos_ids = [1000 + i for i in range(num_alunos)]
    
    # 3. Processar alunos em paralelo
    sucessos, falhas = processar_alunos_paralelo(alunos_ids, dias, pasta_alunos, max_workers)
    
    # 4. Resumo final
    print(f"\n{'='*60}")
    print(f"🎉 TESTE COMPLETO ASSÍNCRONO FINALIZADO")
    print(f"{'='*60}")
    print(f"📊 Resultados:")
    print(f"   ✅ Sucessos: {sucessos}")
    print(f"   ❌ Falhas: {falhas}")
    print(f"   📈 Taxa de sucesso: {sucessos/(sucessos+falhas)*100:.1f}%")
    print(f"   🕐 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 5. Mostrar arquivos gerados
    mostrar_arquivos_gerados(num_alunos, pasta_alunos)
    
    return sucessos > 0

def mostrar_arquivos_gerados(num_alunos: int, pasta_alunos: str):
    """
    Mostra resumo dos arquivos gerados
    """
    print(f"\n📁 Arquivos gerados:")
    
    # Verificar pastas
    pastas = [
        pasta_alunos,
        "output",
        "resultados_ranker_teste",
        "resultados_irt_teste"
    ]
    
    for pasta in pastas:
        if os.path.exists(pasta):
            arquivos = len([f for f in os.listdir(pasta) if f.endswith(('.csv', '.json'))])
            print(f"   📂 {pasta}: {arquivos} arquivos")
        else:
            print(f"   📂 {pasta}: não encontrada")

def verificar_dependencias() -> bool:
    """
    Verifica se todos os scripts necessários existem
    """
    scripts_necessarios = [
        "gerar_alunos.py",
        "gerar_simulacao.py", 
        "treinar_ranker_incremental.py",
        "treinar_irt_incremental.py"
    ]
    
    print("🔍 Verificando dependências...")
    
    for script in scripts_necessarios:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} - NÃO ENCONTRADO!")
            return False
    
    print("✅ Todas as dependências encontradas!")
    return True

def calcular_workers_otimos(num_alunos: int) -> int:
    """
    Calcula número ótimo de workers baseado no número de alunos
    """
    import multiprocessing
    
    # Usar metade dos cores disponíveis, mas no mínimo 2 e no máximo 8
    cores_disponiveis = multiprocessing.cpu_count()
    workers_sugeridos = max(2, min(8, cores_disponiveis // 2))
    
    # Para poucos alunos, usar menos workers
    if num_alunos <= 5:
        workers_sugeridos = min(workers_sugeridos, 3)
    elif num_alunos <= 10:
        workers_sugeridos = min(workers_sugeridos, 4)
    
    return workers_sugeridos

def main():
    parser = argparse.ArgumentParser(description='Teste completo assíncrono: gerar alunos, simular e treinar modelos')
    parser.add_argument('--alunos', type=int, default=50, help='Número de alunos (padrão: 50)')
    parser.add_argument('--dias', type=int, default=100, help='Dias de simulação (padrão: 100)')
    parser.add_argument('--pasta-alunos', default='alunos_teste', help='Pasta para alunos (padrão: alunos_teste)')
    parser.add_argument('--workers', type=int, help='Número de workers paralelos (padrão: automático)')
    parser.add_argument('--verificar', action='store_true', help='Apenas verificar dependências')
    
    args = parser.parse_args()
    
    if args.verificar:
        verificar_dependencias()
        return
    
    # Verificar dependências antes de executar
    if not verificar_dependencias():
        print("❌ Dependências não encontradas. Execute com --verificar para mais detalhes.")
        return
    
    # Calcular número de workers
    if args.workers:
        max_workers = args.workers
    else:
        max_workers = calcular_workers_otimos(args.alunos)
        print(f"🔧 Workers automáticos calculados: {max_workers}")
    
    # Executar teste completo assíncrono
    sucesso = executar_teste_completo_async(
        num_alunos=args.alunos,
        dias=args.dias,
        pasta_alunos=args.pasta_alunos,
        max_workers=max_workers
    )
    
    if sucesso:
        print("\n🎉 Teste completo assíncrono executado com sucesso!")
        print("📁 Verifique as pastas de resultados para os arquivos gerados.")
    else:
        print("\n❌ Teste completo assíncrono falhou!")
        sys.exit(1)

if __name__ == "__main__":
    main()
