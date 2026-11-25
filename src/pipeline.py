from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TARGET = "Class"


def load_data() -> pd.DataFrame:
    """Baixa a base oficial via UCI e persiste uma cópia local reproduzível."""
    raw_path = DATA_DIR / "raw" / "dry_bean.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)
    dataset = fetch_ucirepo(id=602)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    if TARGET not in df.columns:
        if targets.shape[1] != 1:
            raise ValueError(f"Esperava uma coluna alvo; recebi: {list(targets.columns)}")
        df = df.rename(columns={targets.columns[0]: TARGET})
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    return df


def describe_data(df: pd.DataFrame) -> dict:
    """Gera estatísticas descritivas gerais e arquivos para relatório."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric = df.drop(columns=[TARGET]).describe().T
    numeric.to_csv(OUTPUT_DIR / "estatisticas_descritivas.csv", encoding="utf-8-sig")
    class_counts = df[TARGET].value_counts().rename_axis(TARGET).reset_index(name="quantidade")
    class_counts["proporcao"] = class_counts["quantidade"] / len(df)
    class_counts.to_csv(OUTPUT_DIR / "distribuicao_classes.csv", index=False, encoding="utf-8-sig")
    summary = {
        "linhas": int(df.shape[0]),
        "colunas_incluindo_alvo": int(df.shape[1]),
        "atributos": int(df.shape[1] - 1),
        "valores_ausentes": int(df.isna().sum().sum()),
        "linhas_duplicadas": int(df.duplicated().sum()),
        "classes": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "tipos": {str(k): str(v) for k, v in df.dtypes.items()},
    }
    (OUTPUT_DIR / "resumo_base.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def prepare_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicatas exatas antes de criar subconjuntos."""
    before = len(df)
    clean = df.drop_duplicates().reset_index(drop=True)
    return clean, before - len(clean)


def make_splits(df: pd.DataFrame):
    """Cria divisões estratificadas 70/15/15 para treino/validação/teste."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models() -> dict[str, Pipeline]:
    """O scaler é ajustado apenas dentro do treino em cada pipeline."""
    return {
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def run_pipeline() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df_original = load_data()
    stats = describe_data(df_original)
    df, removed = prepare_rows(df_original)
    if df.isna().any().any():
        raise ValueError("Há valores ausentes; defina o tratamento antes do treinamento.")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    split_summary = {
        "treino": len(X_train), "validacao": len(X_val), "teste": len(X_test),
        "duplicatas_removidas": removed,
        "fracao": {"treino": 0.70, "validacao": 0.15, "teste": 0.15},
        "semente": RANDOM_STATE,
    }
    (OUTPUT_DIR / "divisao_dados.json").write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    parts = []
    for name, features, labels_part in [("treino", X_train, y_train), ("validacao", X_val, y_val), ("teste", X_test, y_test)]:
        part = features.copy()
        part[TARGET] = labels_part
        part["particao"] = name
        parts.append(part)
    pd.concat(parts).to_csv(DATA_DIR / "processed" / "dry_bean_particoes.csv", index=False)
    candidates = build_models()
    val_rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_rows.append({"modelo": name, "acuracia_validacao": accuracy_score(y_val, pred), "f1_macro_validacao": f1_score(y_val, pred, average="macro")})
    val_table = pd.DataFrame(val_rows).sort_values("f1_macro_validacao", ascending=False)
    val_table.to_csv(OUTPUT_DIR / "comparacao_validacao.csv", index=False, encoding="utf-8-sig")
    best_name = str(val_table.iloc[0]["modelo"])

    # Com a seleção feita pela validação, treina o estimador final usando treino + validação.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model = clone(candidates[best_name]).fit(X_trainval, y_trainval)
    test_pred = final_model.predict(X_test)
    labels = sorted(df[TARGET].astype(str).unique())
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    accuracy = float(accuracy_score(y_test, test_pred))
    report = classification_report(y_test, test_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "metricas_por_classe.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUTPUT_DIR / "matriz_confusao.csv", encoding="utf-8-sig")
    metrics = {"modelo_escolhido": best_name, "acuracia_teste": accuracy, "f1_macro_teste": float(f1_score(y_test, test_pred, average="macro")), "relatorio": report}
    (OUTPUT_DIR / "metricas_teste.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title=f"Matriz de confusão — {best_name}", xlabel="Classe prevista", ylabel="Classe real")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "matriz_confusao.png", dpi=160)
    plt.close(fig)

    # Exemplo de inferência real usando uma linha reservada do teste.
    example = X_test.iloc[[0]]
    predicted = str(final_model.predict(example)[0])
    probabilities = final_model.predict_proba(example)[0]
    classes = list(final_model.classes_)
    confidence = float(probabilities[classes.index(predicted)])
    inference = {"linha_teste_indice": int(example.index[0]), "classe_prevista": predicted, "confianca_da_classe_prevista": confidence, "entrada": {str(k): float(v) for k, v in example.iloc[0].items()}}
    (OUTPUT_DIR / "exemplo_predicao.json").write_text(json.dumps(inference, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(final_model, MODEL_DIR / "dry_bean_pipeline.joblib")

    print("=== Estatísticas da base (antes de remover duplicatas) ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Duplicatas exatas removidas: {removed}")
    print("\n=== Comparação na validação ===")
    print(val_table.to_string(index=False))
    print("\n=== Resultado final no teste ===")
    print(f"Modelo: {best_name} | Acurácia: {accuracy:.4f} | F1 macro: {metrics['f1_macro_teste']:.4f}")
    print("\n=== Predição demonstrativa ===")
    print(json.dumps(inference, indent=2, ensure_ascii=False))
    print(f"\nArquivos gerados em: {OUTPUT_DIR}")
    return {"stats": stats, "validation": val_table, "metrics": metrics, "inference": inference}


if __name__ == "__main__":
    run_pipeline()



