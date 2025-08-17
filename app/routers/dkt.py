from fastapi import APIRouter, HTTPException
from app.schemas.dkt import PrevisaoRequest, TreinamentoRequest
from app.cron.dkt import treinar_modelo_dkt
import pandas as pd
import pickle
import torch

dkt_router = APIRouter()

@dkt_router.post("/treinar_modelo/")
async def treinar_modelo(request: TreinamentoRequest):
    aluno_id = request.aluno_id
    resultado = await treinar_modelo_dkt(aluno_id)
    if "erro" in resultado:
        raise HTTPException(status_code=400, detail=resultado["erro"])
    return resultado

@dkt_router.post("/prever_desempenho/")
async def prever_desempenho(request: PrevisaoRequest):
    aluno_id = request.aluno_id
    dados_aluno = request.dados_aluno

    # Carregar o modelo para o aluno
    modelo_caminho = f"modelos/student_{aluno_id}/model.pkl"
    try:
        with open(modelo_caminho, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return {"erro": f"Modelo para o aluno {aluno_id} não encontrado"}

    # Criar DataFrame com os dados do aluno
    df_aluno = pd.DataFrame(dados_aluno, columns=["subclasse", "desempenho"])

    # Características para a previsão (você pode adicionar mais features aqui)
    X = df_aluno[["desempenho"]].values  # Usando o desempenho de cada conteúdo

    # Transformando os dados de entrada para o formato adequado para LSTM
    X = torch.tensor(X, dtype=torch.float32).view(1, -1, 1)  # Adicionando batch_size e seq_len (1, seq_len, input_size)

    # Realizando a previsão
    model.eval()  # Coloca o modelo em modo de avaliação
    with torch.no_grad():
        previsao = model(X)

    # A previsão é um tensor, então podemos pegar o valor com `.item()`
    return {"previsao": previsao.item()}