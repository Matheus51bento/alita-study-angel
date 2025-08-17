from sqlalchemy.future import select
from app.db.database import get_session
from app.models.performance import Performance
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class DKTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DKTModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        output = self.fc(
            lstm_out[:, -1, :]
        )  # Considera apenas o último estado da sequência
        return output


class DKTDataSet(Dataset):
    def __init__(self, sequences, input_size):
        self.sequences = sequences
        self.input_size = input_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Normaliza os dados

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Verificando a estrutura de sequence antes de processar
        print(f"[DEBUG] sequence[{idx}]: {sequence}")

        # Garantir que estamos pegando o desempenho de um único conteúdo
        if isinstance(sequence, list) and len(sequence) == 2:
            # sequence[0] é o nome do conteúdo e sequence[1] é o desempenho
            x = sequence[1]  # Desempenho do conteúdo
            print(f"[DEBUG] x (desempenho do conteúdo): {x}")
            y = sequence[1]  # Vamos usar o desempenho do item atual para o target

            x = self.scaler.fit_transform([[i] for i in x])  # shape (seq_len, 1)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 1)

            return x, torch.tensor(y, dtype=torch.float32)
        else:
            raise ValueError(f"Dados mal formados: {sequence}")


async def organizar_dados_para_dkt(aluno_id: str):
    async for session in get_session():
        result = await session.exec(
            select(Performance).filter(Performance.student_id == aluno_id)
        )
        registros = result.scalars().all()

    if not registros:
        print(f"[XGB] Nenhum dado de desempenho encontrado para o aluno {aluno_id}.")
        return []

    # Organizando os dados em um DataFrame
    df = pd.DataFrame([r.__dict__ for r in registros])
    df = df.drop(columns=["_sa_instance_state", "id"])  # Remover colunas desnecessárias
    df["data"] = df["timestamp"].dt.date
    df["score"] = 1 - (
        (df["desempenho"] + 1) / 2
    )  # Convertendo o desempenho para score

    # Agrupar os dados por aluno e conteúdo (subclasse)
    # Agora estamos pegando os dados apenas de um conteúdo por vez
    grouped = df[["subclasse", "desempenho"]].values.tolist()

    # Verificando a estrutura do grouped antes de retornar
    print(f"[DEBUG] Estrutura do grouped: {grouped[:5]}")  # Exibe os primeiros 5 itens
    return grouped


async def treinar_modelo_dkt(aluno_id: str):
    # Organizar os dados
    grouped = await organizar_dados_para_dkt(aluno_id)
    print(f"[LSTM] Dados organizados para o aluno {aluno_id}: {grouped}")
    if not grouped:
        return {
            "erro": f"Sem dados suficientes para treinar o modelo do aluno {aluno_id}."
        }

    # Definir o modelo LSTM
    input_size = 1
    hidden_size = 64
    output_size = 1
    model = DKTModel(input_size, hidden_size, output_size)

    # Preparando os dados
    train_dataset = DKTDataSet(grouped, input_size)
    print(f"[LSTM] Tamanho do dataset de treinamento: {len(train_dataset)}")
    print(
        f"[LSTM] Exemplo de dado: {train_dataset[0]}"
    )  # Exibe o primeiro item do dataset
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Função de treinamento
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Salvando o modelo
    modelo_caminho = f"modelos/student_{aluno_id}/model.pkl"
    os.makedirs(os.path.dirname(modelo_caminho), exist_ok=True)
    with open(modelo_caminho, "wb") as f:
        pickle.dump(model, f)
    print(f"[LSTM] Modelo treinado e salvo para o aluno {aluno_id}")

    return {"sucesso": f"Modelo treinado e salvo para o aluno {aluno_id}"}
