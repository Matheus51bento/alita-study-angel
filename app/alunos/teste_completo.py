import os
import subprocess
import sys
import time
from datetime import datetime
import argparse

def executar_comando(comando, descricao):
    """
    Executa um comando e mostra o progresso
    """
    print(f"\n🔄 {descricao}...")
    print(f"   Comando: {comando}")
    
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"   ✅ {descricao} concluído com sucesso!")
            return True
        else:
            print(f"   ❌ Erro em {descricao}:")
            print(f"   {resultado.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao executar {descricao}: {e}")
        return False

def gerar_alunos(num_alunos, pasta_saida="alunos_teste"):
    """
    Gera múltiplos alunos
    """
    comando = f"python gerar_alunos.py {num_alunos} --pasta-saida {pasta_saida}"
    return executar_comando(comando, f"Gerando {num_alunos} alunos")

def simular_desempenho_aluno(student_id, dias, pasta_alunos="alunos_teste"):
    """
    Simula desempenho para um aluno específico
    """
    arquivo_csv = f"{pasta_alunos}/aluno_{student_id}.csv"
    comando = f"python gerar_simulacao.py {arquivo_csv} {student_id} --dias {dias}"
    return executar_comando(comando, f"Simulando {dias} dias para aluno {student_id}")

def treinar_ranker_aluno(student_id, pasta_saida="resultados_ranker_teste"):
    """
    Treina ranker para um aluno específico
    """
    pasta_aluno = f"output/aluno_{student_id}"
    comando = f"python treinar_ranker_incremental.py {pasta_aluno} --pasta-saida {pasta_saida}"
    return executar_comando(comando, f"Treinando ranker para aluno {student_id}")

def treinar_irt_aluno(student_id, dias, pasta_alunos="alunos_teste", pasta_saida="resultados_irt_teste"):
    """
    Treina IRT para um aluno específico
    """
    arquivo_csv = f"{pasta_alunos}/aluno_{student_id}.csv"
    comando = f"python treinar_irt_incremental.py {arquivo_csv} --dias {dias} --pasta-saida {pasta_saida}"
    return executar_comando(comando, f"Treinando IRT para aluno {student_id}")

def processar_aluno(student_id, dias, pasta_alunos="alunos_teste"):
    """
    Processa um aluno completo: simulação + treinamento ranker + treinamento IRT
    """
    print(f"\n🎯 Processando aluno {student_id}...")
    
    # 1. Simular desempenho
    if not simular_desempenho_aluno(student_id, dias, pasta_alunos):
        return False
    
    # 2. Treinar ranker
    if not treinar_ranker_aluno(student_id):
        return False
    
    # 3. Treinar IRT
    if not treinar_irt_aluno(student_id, dias, pasta_alunos):
        return False
    
    print(f"✅ Aluno {student_id} processado com sucesso!")
    return True

def executar_teste_completo(num_alunos=50, dias=100, pasta_alunos="alunos_teste"):
    """
    Executa o teste completo para todos os alunos
    """
    print("🚀 INICIANDO TESTE COMPLETO")
    print("=" * 50)
    print(f"📊 Configuração:")
    print(f"   👥 Alunos: {num_alunos}")
    print(f"   📅 Dias de simulação: {dias}")
    print(f"   📁 Pasta de alunos: {pasta_alunos}")
    print(f"   🕐 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Gerar alunos
    if not gerar_alunos(num_alunos, pasta_alunos):
        print("❌ Falha ao gerar alunos. Abortando teste.")
        return False
    
    # 2. Processar cada aluno
    sucessos = 0
    falhas = 0
    
    for i in range(num_alunos):
        student_id = 1000 + i
        
        print(f"\n{'='*60}")
        print(f"📈 Progresso: {i+1}/{num_alunos} ({(i+1)/num_alunos*100:.1f}%)")
        print(f"{'='*60}")
        
        if processar_aluno(student_id, dias, pasta_alunos):
            sucessos += 1
        else:
            falhas += 1
            print(f"⚠️  Aluno {student_id} falhou, continuando com os próximos...")
        
        # Pequena pausa para não sobrecarregar o sistema
        time.sleep(1)
    
    # 3. Resumo final
    print(f"\n{'='*60}")
    print(f"🎉 TESTE COMPLETO FINALIZADO")
    print(f"{'='*60}")
    print(f"📊 Resultados:")
    print(f"   ✅ Sucessos: {sucessos}")
    print(f"   ❌ Falhas: {falhas}")
    print(f"   📈 Taxa de sucesso: {sucessos/(sucessos+falhas)*100:.1f}%")
    print(f"   🕐 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 4. Mostrar arquivos gerados
    mostrar_arquivos_gerados(num_alunos, pasta_alunos)
    
    return sucessos > 0

def mostrar_arquivos_gerados(num_alunos, pasta_alunos):
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

def verificar_dependencias():
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

def main():
    parser = argparse.ArgumentParser(description='Teste completo: gerar alunos, simular e treinar modelos')
    parser.add_argument('--alunos', type=int, default=50, help='Número de alunos (padrão: 50)')
    parser.add_argument('--dias', type=int, default=100, help='Dias de simulação (padrão: 100)')
    parser.add_argument('--pasta-alunos', default='alunos_teste', help='Pasta para alunos (padrão: alunos_teste)')
    parser.add_argument('--verificar', action='store_true', help='Apenas verificar dependências')
    
    args = parser.parse_args()
    
    if args.verificar:
        verificar_dependencias()
        return
    
    # Verificar dependências antes de executar
    if not verificar_dependencias():
        print("❌ Dependências não encontradas. Execute com --verificar para mais detalhes.")
        return
    
    # Executar teste completo
    sucesso = executar_teste_completo(
        num_alunos=args.alunos,
        dias=args.dias,
        pasta_alunos=args.pasta_alunos
    )
    
    if sucesso:
        print("\n🎉 Teste completo executado com sucesso!")
        print("📁 Verifique as pastas de resultados para os arquivos gerados.")
    else:
        print("\n❌ Teste completo falhou!")
        sys.exit(1)

if __name__ == "__main__":
    main() 