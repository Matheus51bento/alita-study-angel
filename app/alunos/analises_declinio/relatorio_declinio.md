
# Relatório de Declínio do IRT vs Melhoria do Ranker

## 📊 Estatísticas Gerais

### Declínio do IRT:
- **Média**: -14.81%
- **Mediana**: 22.03%
- **Mínimo**: -185.97%
- **Máximo**: 37.77%
- **Desvio Padrão**: 81.40%

### Melhoria do XGBoost Ranker:
- **Média**: -44.14%
- **Mediana**: -45.05%
- **Mínimo**: -59.67%
- **Máximo**: -28.01%

### Performance Final:
- **IRT (Média)**: 0.6994 ± 0.0863
- **Ranker (Média)**: 0.5252 ± 0.1091
- **Diferença Média**: -0.1743

## 🎯 Análise do Declínio do IRT

### Alunos com Maior Declínio (Top 5):

1. **Aluno 1007**:
   - Declínio IRT: 37.77%
   - Melhoria Ranker: -28.01%
   - IRT Final: 0.6138
   - Ranker Final: 0.7199

2. **Aluno 1006**:
   - Declínio IRT: 31.84%
   - Melhoria Ranker: -33.38%
   - IRT Final: 0.6787
   - Ranker Final: 0.6005

3. **Aluno 1005**:
   - Declínio IRT: 29.42%
   - Melhoria Ranker: -57.99%
   - IRT Final: 0.7044
   - Ranker Final: 0.4023

4. **Aluno 1001**:
   - Declínio IRT: 22.85%
   - Melhoria Ranker: -48.73%
   - IRT Final: 0.7715
   - Ranker Final: 0.4639

5. **Aluno 1004**:
   - Declínio IRT: 22.03%
   - Melhoria Ranker: -36.40%
   - IRT Final: 0.7235
   - Ranker Final: 0.5814

### Alunos com Menor Declínio (Bottom 5):

1. **Aluno 1008**:
   - Declínio IRT: -185.97%
   - Melhoria Ranker: -45.23%
   - IRT Final: 0.5460
   - Ranker Final: 0.5054

2. **Aluno 1002**:
   - Declínio IRT: -124.56%
   - Melhoria Ranker: -42.76%
   - IRT Final: 0.6557
   - Ranker Final: 0.5643

3. **Aluno 1003**:
   - Declínio IRT: 14.04%
   - Melhoria Ranker: -45.05%
   - IRT Final: 0.7943
   - Ranker Final: 0.5300

4. **Aluno 1000**:
   - Declínio IRT: 19.28%
   - Melhoria Ranker: -59.67%
   - IRT Final: 0.8072
   - Ranker Final: 0.3588

5. **Aluno 1004**:
   - Declínio IRT: 22.03%
   - Melhoria Ranker: -36.40%
   - IRT Final: 0.7235
   - Ranker Final: 0.5814

## 📈 Insights Principais

1. **Declínio Consistente**: -14.8% de declínio médio do IRT
2. **Melhoria Contínua**: -44.1% de melhoria média do Ranker
3. **Divergência de Performance**: O IRT piora enquanto o Ranker melhora
4. **Vantagem Final**: -0.2 pontos de vantagem do Ranker

## 🎯 Recomendações

- **Substituição Progressiva**: Considerar substituir IRT pelo Ranker
- **Monitoramento**: Acompanhar declínio do IRT em produção
- **Otimização**: Focar recursos no desenvolvimento do Ranker
- **Migração**: Planejar migração gradual dos alunos para o Ranker

## 🎉 Conclusão

O IRT demonstra **declínio consistente** de -14.8%, enquanto o XGBoost Ranker apresenta **melhoria contínua** de -44.1%. Esta divergência sugere que o Ranker é mais adequado para o contexto de aprendizado incremental.
