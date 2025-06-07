import json
from typing import List, Dict
import pandas as pd

def df_to_json(df: pd.DataFrame, student_id: str) -> List[Dict]:
    """
    Converte um DataFrame para JSON no formato especificado.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        student_id (str): ID do estudante a ser incluído em cada registro
    
    Returns:
        List[Dict]: Lista de dicionários no formato JSON desejado
    """
    result = []
    for _, row in df.iterrows():
        result.append({
            "student_id": student_id,
            "classe": row["classe"],
            "subclasse": row["subclasse"],
            "desempenho": float(row["desempenho"]),
            "peso_classe": float(row["peso_classe"]),
            "peso_subclasse": float(row["peso_subclasse"]),
            "peso_por_questao": float(row["peso_por_questao"])
        })
    
    return result

# Exemplo de uso:
if __name__ == "__main__":
    # Lê o CSV
    data = pd.read_csv("aluno2.csv")
    
    # Chama a função com student_id específico
    json_output = df_to_json(data, "12345")
    
    # Imprime o JSON formatado
    print(json.dumps(json_output, indent=2, ensure_ascii=False))

    # Salva o JSON em um arquivo
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)