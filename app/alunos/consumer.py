import os
import json
import asyncio
import httpx
import traceback

API_URL = "http://localhost:8000/api/v1/performance/batch/"  # ajuste se necessário
PASTA_ARQUIVOS = "./alunos/output/aluno"  # pasta onde estão os arquivos

async def enviar_arquivo_para_api(caminho_arquivo):
    async with httpx.AsyncClient(timeout=30) as client:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        try:
            response = await client.post(API_URL, json=dados)
            response.raise_for_status()
            print(f"[✓] Enviado com sucesso: {caminho_arquivo}")
        except httpx.HTTPStatusError as e:
            print(f"[✗] Erro na resposta da API para {caminho_arquivo}: {e.response.status_code} {e.response.text}")
        except Exception as e:
            print(f"[✗] Erro ao enviar {caminho_arquivo}:")
            traceback.print_exc()

async def main():
    arquivos = [f for f in os.listdir(PASTA_ARQUIVOS) if f.endswith(".json")]
    
    for arquivo in arquivos:
        caminho = os.path.join(PASTA_ARQUIVOS, arquivo)
        await enviar_arquivo_para_api(caminho)

# async def enviar_arquivo_para_api(caminho_arquivo):
#     async with httpx.AsyncClient(timeout=30) as client:
#         with open(caminho_arquivo, "r", encoding="utf-8") as f:
#             dados = json.load(f)

#         try:
#             response = await client.post(API_URL, json=dados)
#             response.raise_for_status()
#             print(f"[✓] Enviado com sucesso: {caminho_arquivo}")
#         except httpx.HTTPStatusError as e:
#             print(f"[✗] Erro na resposta da API para {caminho_arquivo}: {e.response.status_code} {e.response.text}")
#         except Exception as e:
#             print(f"[✗] Erro ao enviar {caminho_arquivo}:")
#             traceback.print_exc()

# async def main():
#     arquivos = [f for f in os.listdir(PASTA_ARQUIVOS) if f.endswith(".json")]
#     tarefas = [
#         enviar_arquivo_para_api(os.path.join(PASTA_ARQUIVOS, arquivo))
#         for arquivo in arquivos
#     ]
#     await asyncio.gather(*tarefas)

if __name__ == "__main__":
    asyncio.run(main())
