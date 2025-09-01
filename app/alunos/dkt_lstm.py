import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, roc_curve
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configurar seed para reprodutibilidade
torch.manual_seed(42)
np.random.seed(42)

class DKTDataset(Dataset):
    """
    Dataset para DKT multitarefa com alvos de regressão e classificação
    """
    def __init__(self, sequences: List[np.ndarray], targets_reg: List[float], targets_cls: List[int]):
        self.sequences = sequences
        self.targets_reg = targets_reg  # float - valores contínuos
        self.targets_cls = targets_cls  # int - 0/1
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.sequences[idx])
        y_reg = torch.FloatTensor([self.targets_reg[idx]])
        y_cls = torch.FloatTensor([self.targets_cls[idx]])
        return x, y_reg.squeeze(), y_cls.squeeze()

class DKTModel(nn.Module):
    """
    Modelo DKT multitarefa usando LSTM com dois heads: regressão e classificação
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3):
        super(DKTModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM compartilhado
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout compartilhado
        self.dropout = nn.Dropout(dropout)
        
        # Heads separados
        self.fc_reg = nn.Linear(hidden_size, 1)   # regressão (contínuo)
        self.fc_cls = nn.Linear(hidden_size, 1)   # classificação (logit)
        
    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Pegar apenas o último output da sequência
        h_last = lstm_out[:, -1, :]
        h_last = self.dropout(h_last)
        
        # Head de regressão (com limitação a 1.0)
        y_reg = self.fc_reg(h_last).squeeze(-1)
        y_reg = torch.clamp(y_reg, -1.0, 1.0)  # Limitar entre -1 e 1
        
        # Head de classificação (logit + sigmoid)
        y_logit = self.fc_cls(h_last).squeeze(-1)
        y_prob = torch.sigmoid(y_logit)
        
        return y_reg, y_logit, y_prob

def carregar_dados_aluno(student_id: int, dias: int = 20, pasta_output: str = "output") -> List[Dict]:
    """
    Carrega dados de um aluno específico
    """
    pasta_aluno = os.path.join(pasta_output, f"aluno_{student_id}")
    
    if not os.path.exists(pasta_aluno):
        raise FileNotFoundError(f"Pasta do aluno {student_id} não encontrada: {pasta_aluno}")
    
    dados = []
    for dia in range(1, dias + 1):
        arquivo = os.path.join(pasta_aluno, f"desempenho_dia_{dia}.json")
        
        if not os.path.exists(arquivo):
            print(f"⚠️  Arquivo do dia {dia} não encontrado, parando em {len(dados)} dias")
            break
        
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados_dia = json.load(f)
            dados.append(dados_dia)
    
    print(f"📊 Carregados {len(dados)} dias de dados para aluno {student_id}")
    return dados

def preparar_dados_dkt(dados: List[Dict], window_size: int = 5, dias_treino: int = 60, normalizar: bool = True) -> Tuple[List[np.ndarray], List[float], List[int], List[np.ndarray], List[float], List[int], Dict]:
    """
    Prepara dados para DKT criando sequências temporais
    Separa dados de treino (60 dias) e teste (5 dias)
    """
    print("🔄 Preparando dados para DKT...")
    
    # Separar dados de treino e teste
    dados_treino = dados[:dias_treino]
    dados_teste = dados[dias_treino:]
    
    # Calcular estatísticas para normalização (apenas com dados de treino)
    if normalizar:
        print("   🔧 Calculando estatísticas para normalização...")
        todas_features_treino = []
        for dados_dia in dados_treino:
            for item in dados_dia:
                feature_vector = [
                    item['desempenho'],
                    item['peso_classe'],
                    item['peso_subclasse'], 
                    item['peso_por_questao']
                ]
                todas_features_treino.append(feature_vector)
        
        todas_features_treino = np.array(todas_features_treino)
        feature_means = np.mean(todas_features_treino, axis=0)
        feature_stds = np.std(todas_features_treino, axis=0)
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # Evitar divisão por zero
        
        normalizacao_stats = {
            'means': feature_means.tolist(),
            'stds': feature_stds.tolist()
        }
        print(f"   📊 Estatísticas de normalização calculadas")
    else:
        normalizacao_stats = None
    
    print(f"   📊 Dados de treino: {len(dados_treino)} dias")
    print(f"   📊 Dados de teste: {len(dados_teste)} dias")
    print(f"   📊 Configuração: {dias_treino} dias treino + {len(dados_teste)} dias teste = {len(dados)} dias total")
    
    # Preparar dados de treino
    features_treino = []
    targets_reg_treino = []  # Alvos de regressão (contínuos)
    targets_cls_treino = []  # Alvos de classificação (binários)
    
    for dia_idx, dados_dia in enumerate(dados_treino):
        # Features do dia atual
        features_dia = []
        
        for item in dados_dia:
            # Features: desempenho, peso_classe, peso_subclasse, peso_por_questao
            feature_vector = [
                item['desempenho'],
                item['peso_classe'],
                item['peso_subclasse'], 
                item['peso_por_questao']
            ]
            
            # Aplicar normalização se habilitada
            if normalizar and normalizacao_stats is not None:
                feature_vector = (np.array(feature_vector) - normalizacao_stats['means']) / normalizacao_stats['stds']
            
            features_dia.append(feature_vector)
        
        # Converter para array
        features_array = np.array(features_dia)
        features_treino.append(features_array)
        
        # Target de regressão: média do desempenho do próximo dia (se existir)
        if dia_idx < len(dados_treino) - 1:
            # Calcular target baseado no próximo dia
            proximo_dia = dados_treino[dia_idx + 1]
            desempenhos_proximo = [item['desempenho'] for item in proximo_dia]
            target_reg = np.mean(desempenhos_proximo)
        else:
            # Para o último dia, usar o próprio desempenho médio
            desempenhos_atual = [item['desempenho'] for item in dados_dia]
            target_reg = np.mean(desempenhos_atual)
        
        # Target de classificação: se o próximo dia tem desempenho > percentil 60 (critério mais balanceado)
        if dia_idx < len(dados_treino) - 1:
            proximo_dia = dados_treino[dia_idx + 1]
            desempenhos_proximo = [item['desempenho'] for item in proximo_dia]
            media_proximo = np.mean(desempenhos_proximo)
            
            # Calcular percentil 60 de todos os dias para usar como threshold
            todas_medias = []
            for dados_dia_geral in dados_treino:
                desempenhos_dia = [item['desempenho'] for item in dados_dia_geral]
                todas_medias.append(np.mean(desempenhos_dia))
            threshold_percentil = np.percentile(todas_medias, 60)
            
            target_cls = 1 if media_proximo > threshold_percentil else 0
        else:
            desempenhos_atual = [item['desempenho'] for item in dados_dia]
            media_atual = np.mean(desempenhos_atual)
            
            # Calcular percentil 60 de todos os dias para usar como threshold
            todas_medias = []
            for dados_dia_geral in dados_treino:
                desempenhos_dia = [item['desempenho'] for item in dados_dia_geral]
                todas_medias.append(np.mean(desempenhos_dia))
            threshold_percentil = np.percentile(todas_medias, 60)
            
            target_cls = 1 if media_atual > threshold_percentil else 0
        
        targets_reg_treino.append(target_reg)
        targets_cls_treino.append(target_cls)
    
    # Criar sequências de treino com janela deslizante
    sequences_treino = []
    targets_reg_sequences = []
    targets_cls_sequences = []
    
    for i in range(len(features_treino) - window_size + 1):
        # Sequência de features
        seq = features_treino[i:i + window_size]
        # Padronizar tamanho da sequência (média dos conteúdos por dia)
        seq_padded = []
        for dia_features in seq:
            # Usar média das features do dia
            dia_avg = np.mean(dia_features, axis=0)
            seq_padded.append(dia_avg)
        
        sequences_treino.append(np.array(seq_padded))
        targets_reg_sequences.append(targets_reg_treino[i + window_size - 1])
        targets_cls_sequences.append(targets_cls_treino[i + window_size - 1])
    
    # Preparar dados de teste (usar última sequência de treino para prever próximos dias)
    features_teste = []
    targets_reg_teste = []
    targets_cls_teste = []
    
    for dia_idx, dados_dia in enumerate(dados_teste):
        # Features do dia atual
        features_dia = []
        
        for item in dados_dia:
            feature_vector = [
                item['desempenho'],
                item['peso_classe'],
                item['peso_subclasse'], 
                item['peso_por_questao']
            ]
            
            # Aplicar normalização se habilitada (usando estatísticas do treino)
            if normalizar and normalizacao_stats is not None:
                feature_vector = (np.array(feature_vector) - normalizacao_stats['means']) / normalizacao_stats['stds']
            
            features_dia.append(feature_vector)
        
        features_array = np.array(features_dia)
        features_teste.append(features_array)
        
        # Target de regressão: média do desempenho do dia atual
        desempenhos_atual = [item['desempenho'] for item in dados_dia]
        target_reg = np.mean(desempenhos_atual)
        
        # Target de classificação: se o dia atual tem desempenho > percentil 60 (critério mais balanceado)
        # Calcular percentil 60 de todos os dias de treino para usar como threshold
        todas_medias_treino = []
        for dados_dia_geral in dados_treino:
            desempenhos_dia = [item['desempenho'] for item in dados_dia_geral]
            todas_medias_treino.append(np.mean(desempenhos_dia))
        threshold_percentil = np.percentile(todas_medias_treino, 60)
        
        target_cls = 1 if target_reg > threshold_percentil else 0
        
        targets_reg_teste.append(target_reg)
        targets_cls_teste.append(target_cls)
    
    print(f"   📊 Criadas {len(sequences_treino)} sequências de treino com janela de {window_size} dias")
    print(f"   🎯 Features por sequência: {sequences_treino[0].shape}")
    print(f"   📊 Dados de teste: {len(targets_reg_teste)} dias para previsão (limitado a 5 dias)")
    
    # Mostrar distribuição das classes
    targets_cls_array = np.array(targets_cls_sequences)
    targets_cls_teste_array = np.array(targets_cls_teste)
    
    print(f"   📈 Distribuição das classes:")
    print(f"      🎯 Treino - Classe 0: {np.sum(targets_cls_array == 0)} ({np.mean(targets_cls_array == 0)*100:.1f}%)")
    print(f"      🎯 Treino - Classe 1: {np.sum(targets_cls_array == 1)} ({np.mean(targets_cls_array == 1)*100:.1f}%)")
    print(f"      🔮 Teste - Classe 0: {np.sum(targets_cls_teste_array == 0)} ({np.mean(targets_cls_teste_array == 0)*100:.1f}%)")
    print(f"      🔮 Teste - Classe 1: {np.sum(targets_cls_teste_array == 1)} ({np.mean(targets_cls_teste_array == 1)*100:.1f}%)")
    
    return sequences_treino, targets_reg_sequences, targets_cls_sequences, features_teste, targets_reg_teste, targets_cls_teste, normalizacao_stats

def treinar_modelo_dkt(sequences: List[np.ndarray], targets_reg: List[float], targets_cls: List[int],
                      input_size: int = 4, hidden_size: int = 64, num_layers: int = 2,
                      learning_rate: float = 0.001, epochs: int = 100, 
                      batch_size: int = 8, alpha: float = 1.0, beta: float = 0.7,
                      patience: int = 10, val_split: float = 0.2) -> Tuple[DKTModel, List[float]]:
    """
    Treina o modelo DKT
    """
    print("🚀 Treinando modelo DKT...")
    
    # Criar dataset e dataloader
    dataset = DKTDataset(sequences, targets_reg, targets_cls)
    
    # Split treino/validação
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   📊 Split treino/validação: {train_size}/{val_size} amostras")
    
    # Inicializar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   🔧 Usando dispositivo: {device}")
    
    model = DKTModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()  # Para classificação (usa logit)
    mse_loss = nn.MSELoss()  # Para regressão
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"   🎯 Perda combinada: α={alpha}*BCE + β={beta}*MSE")
    
    # Treinamento com early stopping
    losses = []
    val_aucs = []
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"   🎯 Early stopping: patience={patience}, monitorando AUC de validação")
    
    for epoch in range(epochs):
        # Treinamento
        model.train()
        epoch_loss = 0.0
        
        for batch_sequences, batch_targets_reg, batch_targets_cls in train_dataloader:
            batch_sequences = batch_sequences.to(device)
            batch_targets_reg = batch_targets_reg.to(device)
            batch_targets_cls = batch_targets_cls.to(device)
            
            # Forward pass
            y_reg_hat, y_logit, y_prob = model(batch_sequences)
            
            # Calcular perdas
            loss_cls = bce_loss(y_logit, batch_targets_cls)
            loss_reg = mse_loss(y_reg_hat, batch_targets_reg)
            
            # Perda combinada
            loss = alpha * loss_cls + beta * loss_reg
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        losses.append(avg_loss)
        
        # Validação
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets_reg, batch_targets_cls in val_dataloader:
                batch_sequences = batch_sequences.to(device)
                batch_targets_cls = batch_targets_cls.to(device)
                
                y_reg_hat, y_logit, y_prob = model(batch_sequences)
                val_predictions.extend(y_prob.cpu().numpy())
                val_targets.extend(batch_targets_cls.cpu().numpy())
        
        # Calcular AUC de validação
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_targets, val_predictions)
        except ValueError:
            val_auc = 0.5  # Valor neutro se há apenas uma classe
        
        val_aucs.append(val_auc)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"   📈 Época {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Parar se não melhorou por 'patience' épocas
        if patience_counter >= patience:
            print(f"   🛑 Early stopping na época {epoch + 1} (melhor Val AUC: {best_val_auc:.4f})")
            break
    
    # Carregar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   ✅ Melhor modelo carregado (Val AUC: {best_val_auc:.4f})")
    
    print(f"✅ Treinamento concluído!")
    return model, losses

def avaliar_modelo_dkt(model: DKTModel, sequences_treino: List[np.ndarray], targets_reg_treino: List[float], targets_cls_treino: List[int],
                      features_teste: List[np.ndarray], targets_reg_teste: List[float], targets_cls_teste: List[int],
                      batch_size: int = 8, calibrar: bool = True) -> Dict[str, float]:
    """
    Avalia o modelo DKT e faz previsões futuras
    """
    print("📊 Avaliando modelo...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 1. Avaliar no conjunto de treino
    dataset_treino = DKTDataset(sequences_treino, targets_reg_treino, targets_cls_treino)
    dataloader_treino = DataLoader(dataset_treino, batch_size=batch_size, shuffle=False)
    
    all_predictions_reg_treino = []
    all_predictions_cls_treino = []
    all_targets_reg_treino = []
    all_targets_cls_treino = []
    
    with torch.no_grad():
        for batch_sequences, batch_targets_reg, batch_targets_cls in dataloader_treino:
            batch_sequences = batch_sequences.to(device)
            
            y_reg_hat, y_logit, y_prob = model(batch_sequences)
            
            all_predictions_reg_treino.extend(y_reg_hat.cpu().numpy())
            all_predictions_cls_treino.extend(y_prob.cpu().numpy())
            all_targets_reg_treino.extend(batch_targets_reg.numpy())
            all_targets_cls_treino.extend(batch_targets_cls.numpy())
    
    # 2. Fazer previsões futuras
    print("🔮 Fazendo previsões futuras...")
    
    # Usar a última sequência de treino para prever os próximos dias
    ultima_sequencia = sequences_treino[-1]  # Última sequência de treino
    
    previsoes_reg_futuras = []
    previsoes_cls_futuras = []
    sequencia_atual = ultima_sequencia.copy()
    
    for dia_teste in range(len(targets_reg_teste)):
        # Converter para tensor
        input_tensor = torch.FloatTensor(sequencia_atual).unsqueeze(0).to(device)
        
        # Fazer previsão
        with torch.no_grad():
            y_reg_hat, y_logit, y_prob = model(input_tensor)
            previsao_reg = y_reg_hat.cpu().numpy()
            previsao_cls = y_prob.cpu().numpy()
        
        previsoes_reg_futuras.append(previsao_reg)
        previsoes_cls_futuras.append(previsao_cls)
        
        # Atualizar sequência (remover primeiro dia, adicionar previsão)
        # Para simplificar, vamos usar a média das features do dia atual
        features_dia_atual = features_teste[dia_teste]
        media_features = np.mean(features_dia_atual, axis=0)
        
        # Atualizar sequência
        sequencia_atual = np.vstack([sequencia_atual[1:], media_features.reshape(1, -1)])
    
    # Converter para arrays
    y_pred_reg_treino = np.array(all_predictions_reg_treino)
    y_true_reg_treino = np.array(all_targets_reg_treino)
    y_pred_cls_treino = np.array(all_predictions_cls_treino)
    y_true_cls_treino = np.array(all_targets_cls_treino)
    
    y_pred_reg_futuro = np.array(previsoes_reg_futuras)
    y_true_reg_futuro = np.array(targets_reg_teste)
    y_pred_cls_futuro = np.array(previsoes_cls_futuras)
    y_true_cls_futuro = np.array(targets_cls_teste)
    
    # Calibração das probabilidades (Platt Scaling)
    if calibrar:
        print("   🔧 Aplicando calibração Platt Scaling...")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            
            # Treinar calibrador com dados de treino
            base_classifier = LogisticRegression(random_state=42)
            base_classifier.fit(y_pred_cls_treino.reshape(-1, 1), y_true_cls_treino)
            
            calibrador = CalibratedClassifierCV(
                base_classifier, 
                cv='prefit', 
                method='sigmoid'
            )
            
            # Reshape para o formato esperado
            X_calib = y_pred_cls_treino.reshape(-1, 1)
            y_calib = y_true_cls_treino
            
            # Treinar calibrador
            calibrador.fit(X_calib, y_calib)
            
            # Aplicar calibração
            y_pred_cls_treino_calib = calibrador.predict_proba(X_calib)[:, 1]
            y_pred_cls_futuro_calib = calibrador.predict_proba(y_pred_cls_futuro.reshape(-1, 1))[:, 1]
            
            # Usar probabilidades calibradas
            y_pred_cls_treino = y_pred_cls_treino_calib
            y_pred_cls_futuro = y_pred_cls_futuro_calib
            
            print("   ✅ Calibração aplicada com sucesso")
            
        except Exception as e:
            print(f"   ⚠️  Erro na calibração: {e}. Usando probabilidades originais.")
    else:
        print("   ⏭️  Calibração desabilitada")
    
    # Calcular métricas de regressão
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Métricas de regressão - treino
    mse_treino = mean_squared_error(y_true_reg_treino, y_pred_reg_treino)
    mae_treino = mean_absolute_error(y_true_reg_treino, y_pred_reg_treino)
    r2_treino = r2_score(y_true_reg_treino, y_pred_reg_treino)
    
    # Métricas de regressão - previsão futura
    mse_futuro = mean_squared_error(y_true_reg_futuro, y_pred_reg_futuro)
    mae_futuro = mean_absolute_error(y_true_reg_futuro, y_pred_reg_futuro)
    r2_futuro = r2_score(y_true_reg_futuro, y_pred_reg_futuro)
    
    # Calcular métricas de classificação (alvos distintos)
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
    
    # Métricas de classificação - treino
    try:
        auc_treino = roc_auc_score(y_true_cls_treino, y_pred_cls_treino)
    except ValueError:
        auc_treino = 0.5  # Valor neutro se há apenas uma classe
    
    try:
        accuracy_treino = accuracy_score(y_true_cls_treino, y_pred_cls_treino > 0.5)
    except ValueError:
        accuracy_treino = 1.0  # Se há apenas uma classe, accuracy é 1.0
    
    try:
        logloss_treino = log_loss(y_true_cls_treino, y_pred_cls_treino)
    except ValueError:
        logloss_treino = 0.0  # Se há apenas uma classe, logloss é 0.0
    
    # Métricas de classificação - futuro
    try:
        auc_futuro = roc_auc_score(y_true_cls_futuro, y_pred_cls_futuro)
    except ValueError:
        auc_futuro = 0.5  # Valor neutro se há apenas uma classe
    
    try:
        accuracy_futuro = accuracy_score(y_true_cls_futuro, y_pred_cls_futuro > 0.5)
    except ValueError:
        accuracy_futuro = 1.0  # Se há apenas uma classe, accuracy é 1.0
    
    try:
        logloss_futuro = log_loss(y_true_cls_futuro, y_pred_cls_futuro)
    except ValueError:
        logloss_futuro = 0.0  # Se há apenas uma classe, logloss é 0.0
    
    metrics = {
        # Métricas de regressão
        'mse_treino': mse_treino,
        'mae_treino': mae_treino,
        'r2_treino': r2_treino,
        'mse_futuro': mse_futuro,
        'mae_futuro': mae_futuro,
        'r2_futuro': r2_futuro,
        # Métricas de classificação
        'auc_treino': auc_treino,
        'accuracy_treino': accuracy_treino,
        'logloss_treino': logloss_treino,
        'auc_futuro': auc_futuro,
        'accuracy_futuro': accuracy_futuro,
        'logloss_futuro': logloss_futuro
    }
    
    print(f"   📈 Regressão - Treino: MSE={mse_treino:.4f}, MAE={mae_treino:.4f}, R²={r2_treino:.4f}")
    print(f"   🔮 Regressão - Futuro: MSE={mse_futuro:.4f}, MAE={mae_futuro:.4f}, R²={r2_futuro:.4f}")
    print(f"   🎯 Classificação - Treino: AUC={auc_treino:.4f}, Accuracy={accuracy_treino:.4f}, LogLoss={logloss_treino:.4f}")
    print(f"   🎯 Classificação - Futuro: AUC={auc_futuro:.4f}, Accuracy={accuracy_futuro:.4f}, LogLoss={logloss_futuro:.4f}")
    
    return metrics, y_pred_reg_treino, y_true_reg_treino, y_pred_cls_treino, y_true_cls_treino, y_pred_reg_futuro, y_true_reg_futuro, y_pred_cls_futuro, y_true_cls_futuro

def plotar_resultados(losses: List[float], y_pred_reg_treino: np.ndarray, y_true_reg_treino: np.ndarray,
                     y_pred_cls_treino: np.ndarray, y_true_cls_treino: np.ndarray,
                     y_pred_reg_futuro: np.ndarray, y_true_reg_futuro: np.ndarray,
                     y_pred_cls_futuro: np.ndarray, y_true_cls_futuro: np.ndarray,
                     metrics: Dict[str, float], pasta_saida: str = "resultados_dkt"):
    """
    Plota resultados do treinamento, avaliação e previsões futuras
    """
    print("📊 Gerando gráficos...")
    
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Criar figura com subplots (2x3 para acomodar mais gráficos)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('🧠 DKT - Resultados de Treinamento e Previsões', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Loss durante treinamento
    ax1.plot(losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss durante Treinamento')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Predições vs Valores Reais (Regressão - Treino)
    ax2.scatter(y_true_reg_treino, y_pred_reg_treino, alpha=0.7, color='#FF6B6B', label='Regressão')
    ax2.plot([y_true_reg_treino.min(), y_true_reg_treino.max()], [y_true_reg_treino.min(), y_true_reg_treino.max()], 'k--', alpha=0.5)
    ax2.set_xlabel('Valor Real')
    ax2.set_ylabel('Predição')
    ax2.set_title(f'Regressão - Treino (R² = {metrics["r2_treino"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Previsões Futuras vs Valores Reais (Regressão)
    dias_futuros = list(range(1, len(y_pred_reg_futuro) + 1))
    ax3.plot(dias_futuros, y_true_reg_futuro, 'o-', linewidth=2, markersize=8, 
             color='#4ECDC4', label='Valor Real', alpha=0.8)
    ax3.plot(dias_futuros, y_pred_reg_futuro, 's-', linewidth=2, markersize=8, 
             color='#FF6B6B', label='Previsão DKT', alpha=0.8)
    ax3.set_xlabel('Dia Futuro')
    ax3.set_ylabel('Desempenho')
    ax3.set_title(f'Regressão - Futuro (R² = {metrics["r2_futuro"]:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Curva ROC - Treino
    try:
        from sklearn.metrics import roc_curve
        fpr_treino, tpr_treino, _ = roc_curve(y_true_cls_treino, y_pred_cls_treino)
        ax4.plot(fpr_treino, tpr_treino, 'r-', linewidth=2, 
                 label=f'Treino (AUC = {metrics["auc_treino"]:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('Curva ROC - Treino')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    except Exception as e:
        ax4.text(0.5, 0.5, f'ROC não disponível\n{str(e)}', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Curva ROC - Treino')
    
    # Gráfico 5: Curva ROC - Futuro
    try:
        fpr_futuro, tpr_futuro, _ = roc_curve(y_true_cls_futuro, y_pred_cls_futuro)
        ax5.plot(fpr_futuro, tpr_futuro, 'b-', linewidth=2, 
                 label=f'Futuro (AUC = {metrics["auc_futuro"]:.3f})')
        ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('Curva ROC - Futuro')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    except Exception as e:
        ax5.text(0.5, 0.5, f'ROC não disponível\n{str(e)}', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Curva ROC - Futuro')
    
    # Gráfico 6: Métricas Comparativas (Regressão + Classificação)
    metric_names = ['R² Treino', 'R² Futuro', 'AUC Treino', 'AUC Futuro', 'Accuracy Treino', 'Accuracy Futuro', 'LogLoss Treino', 'LogLoss Futuro']
    metric_values = [metrics['r2_treino'], metrics['r2_futuro'], 
                    metrics['auc_treino'], metrics['auc_futuro'],
                    metrics['accuracy_treino'], metrics['accuracy_futuro'],
                    metrics['logloss_treino'], metrics['logloss_futuro']]
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#F39C12', '#9B59B6', '#E67E22', '#2ECC71', '#E74C3C']
    
    bars = ax6.bar(metric_names, metric_values, color=colors, alpha=0.7)
    ax6.set_ylabel('Valor')
    ax6.set_title('Métricas Comparativas (Regressão + Classificação)')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, metric_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    nome_arquivo = "resultados_dkt.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Gráficos salvos em: {caminho}")

def salvar_metricas(metrics: Dict[str, float], losses: List[float], 
                   pasta_saida: str = "resultados_dkt"):
    """
    Salva métricas e resultados
    """
    # Converter valores numpy para float para serialização JSON
    metrics_serializable = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # Se é numpy type
            metrics_serializable[key] = float(value.item())
        else:
            metrics_serializable[key] = float(value)
    
    resultados = {
        'metrics': metrics_serializable,
        'final_loss': float(losses[-1]) if losses else 0.0,
        'training_losses': [float(loss) for loss in losses],
        'model_config': {
            'input_size': 4,
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'epochs': 100
        }
    }
    
    nome_arquivo = "metricas_dkt.json"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Métricas salvas em: {caminho}")

def salvar_modelo(model: DKTModel, pasta_saida: str = "resultados_dkt"):
    """
    Salva o modelo treinado
    """
    nome_arquivo = "modelo_dkt.pth"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    
    torch.save(model.state_dict(), caminho)
    print(f"   💾 Modelo salvo em: {caminho}")

def main():
    parser = argparse.ArgumentParser(description='DKT multitarefa com LSTM para prever desempenho (regressão) e acerto (classificação)')
    parser.add_argument('student_id', type=int, help='ID do aluno')
    parser.add_argument('--dias', type=int, default=20, help='Número de dias (padrão: 20)')
    parser.add_argument('--window-size', type=int, default=5, help='Tamanho da janela temporal (padrão: 5)')
    parser.add_argument('--hidden-size', type=int, default=64, help='Tamanho da camada oculta (padrão: 64)')
    parser.add_argument('--num-layers', type=int, default=2, help='Número de camadas LSTM (padrão: 2)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Taxa de aprendizado (padrão: 0.001)')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas (padrão: 100)')
    parser.add_argument('--batch-size', type=int, default=8, help='Tamanho do batch (padrão: 8)')
    parser.add_argument('--pasta-saida', default='resultados_dkt', help='Pasta para salvar resultados')
    parser.add_argument('--patience', type=int, default=10, help='Patience para early stopping (padrão: 10)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Proporção de validação (padrão: 0.2)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Peso da perda de classificação (padrão: 1.0)')
    parser.add_argument('--beta', type=float, default=0.7, help='Peso da perda de regressão (padrão: 0.7)')
    parser.add_argument('--no-calibracao', action='store_true', help='Desabilitar calibração')
    parser.add_argument('--no-normalizacao', action='store_true', help='Desabilitar normalização')
    
    args = parser.parse_args()
    
    print("🧠 DKT MULTITAREFA COM LSTM - REGRESSÃO + CLASSIFICAÇÃO")
    print("=" * 50)
    print(f"👤 Aluno: {args.student_id}")
    print(f"📅 Dias: {args.dias}")
    print(f"🪟 Janela temporal: {args.window_size}")
    print(f"🧠 Hidden size: {args.hidden_size}")
    print(f"📚 Camadas LSTM: {args.num_layers}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🔄 Épocas: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    try:
        # 1. Carregar dados do aluno
        dados = carregar_dados_aluno(args.student_id, args.dias)
        
        # 2. Preparar dados para DKT (65 dias total, treino com 60, teste com 5)
        sequences_treino, targets_reg_treino, targets_cls_treino, features_teste, targets_reg_teste, targets_cls_teste, normalizacao_stats = preparar_dados_dkt(
            dados, args.window_size, dias_treino=60, normalizar=not args.no_normalizacao
        )
        
        if len(sequences_treino) < 2:
            print("❌ Dados insuficientes para treinamento!")
            return
        
        # 3. Treinar modelo
        model, losses = treinar_modelo_dkt(
            sequences_treino, targets_reg_treino, targets_cls_treino,
            input_size=4,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            alpha=args.alpha, beta=args.beta,
            patience=args.patience, val_split=args.val_split
        )
        
        # 4. Avaliar modelo e fazer previsões futuras
        metrics, y_pred_reg_treino, y_true_reg_treino, y_pred_cls_treino, y_true_cls_treino, y_pred_reg_futuro, y_true_reg_futuro, y_pred_cls_futuro, y_true_cls_futuro = avaliar_modelo_dkt(
            model, sequences_treino, targets_reg_treino, targets_cls_treino, features_teste, targets_reg_teste, targets_cls_teste, args.batch_size, calibrar=not args.no_calibracao
        )
        
        # 5. Plotar resultados
        plotar_resultados(losses, y_pred_reg_treino, y_true_reg_treino, y_pred_cls_treino, y_true_cls_treino,
                         y_pred_reg_futuro, y_true_reg_futuro, y_pred_cls_futuro, y_true_cls_futuro, metrics, args.pasta_saida)
        
        # 6. Salvar métricas e modelo
        salvar_metricas(metrics, losses, args.pasta_saida)
        salvar_modelo(model, args.pasta_saida)
        
        print(f"\n🎉 DKT concluído com sucesso!")
        print(f"📁 Resultados salvos em: {args.pasta_saida}")
        print(f"📊 Métricas finais:")
        print(f"   📈 REGRESSÃO - Treino: MSE={metrics['mse_treino']:.4f}, MAE={metrics['mae_treino']:.4f}, R²={metrics['r2_treino']:.4f}")
        print(f"   🔮 REGRESSÃO - Futuro: MSE={metrics['mse_futuro']:.4f}, MAE={metrics['mae_futuro']:.4f}, R²={metrics['r2_futuro']:.4f}")
        print(f"   🎯 CLASSIFICAÇÃO - Treino: AUC={metrics['auc_treino']:.4f}, Accuracy={metrics['accuracy_treino']:.4f}, LogLoss={metrics['logloss_treino']:.4f}")
        print(f"   🎯 CLASSIFICAÇÃO - Futuro: AUC={metrics['auc_futuro']:.4f}, Accuracy={metrics['accuracy_futuro']:.4f}, LogLoss={metrics['logloss_futuro']:.4f}")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 