"""Gera site/data/ai4i.json: tudo o que o dashboard estático precisa, calculado uma vez.

Mesma lógica do dashboard.py (Streamlit): as regras documentadas por Matzka (2020),
as três features derivadas delas e os dois Random Forest multi-label (com e sem essas
features). A página só desenha; nenhum modelo roda no navegador.

    python exportar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

RAIZ = Path(__file__).resolve().parent
MODOS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LIMIAR_OSF = {"L": 11000, "M": 12000, "H": 13000}
FEATURES_BASE = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_encoded",
]
FEATURES_ENG = FEATURES_BASE + ["delta_temp", "power_W", "wear_torque"]


def carregar() -> pd.DataFrame:
    df = pd.read_csv(RAIZ / "ai4i2020.csv", encoding="utf-8-sig")
    df["delta_temp"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["regra_HDF"] = ((df["delta_temp"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)).astype(int)
    df["regra_PWF"] = ((df["power_W"] < 3500) | (df["power_W"] > 9000)).astype(int)
    df["regra_OSF"] = (df["wear_torque"] > df["Type"].map(LIMIAR_OSF)).astype(int)
    df["regra_TWF"] = df["Tool wear [min]"].between(200, 240).astype(int)
    return df


def validar_regras(df: pd.DataFrame) -> list[dict]:
    saida = []
    for modo in ["HDF", "PWF", "OSF", "TWF"]:
        regra, real = df[f"regra_{modo}"], df[modo]
        saida.append(
            {
                "modo": modo,
                "concordancia": round(float((regra == real).mean()), 4),
                "falsos_positivos": int(((regra == 1) & (real == 0)).sum()),
                "falsos_negativos": int(((regra == 0) & (real == 1)).sum()),
                "deterministica": modo != "TWF",
            }
        )
    return saida


def treinar(df: pd.DataFrame) -> dict:
    dm = df.copy()
    dm["Type_encoded"] = LabelEncoder().fit_transform(dm["Type"])
    y = dm[MODOS]
    resultado = {}
    for nome, cols in (("base", FEATURES_BASE), ("eng", FEATURES_ENG)):
        x_tr, x_te, y_tr, y_te = train_test_split(
            dm[cols], y, test_size=0.3, random_state=42, stratify=dm["Machine failure"]
        )
        clf = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        )
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_te)
        resultado[nome] = {
            "f1": {m: round(float(f1_score(y_te.iloc[:, i], pred[:, i], zero_division=0)), 3) for i, m in enumerate(MODOS)},
            "importancia": {
                m: {f: round(float(v), 4) for f, v in zip(cols, clf.estimators_[i].feature_importances_)}
                for i, m in enumerate(MODOS)
            },
        }
    return {
        "f1_base": resultado["base"]["f1"],
        "f1_eng": resultado["eng"]["f1"],
        "importancia_eng": resultado["eng"]["importancia"],
        "teste": {"proporcao": 0.3, "random_state": 42, "estratificado_por": "Machine failure"},
    }


def registros(df: pd.DataFrame) -> dict:
    """Uma linha por máquina, em colunas, para a página filtrar por tipo e desenhar os pontos."""
    bits = sum(df[m].astype(int) * (1 << i) for i, m in enumerate(MODOS))  # modos de falha num inteiro (bit i = MODOS[i])
    return {
        "tipo": df["Type"].tolist(),
        "rpm": df["Rotational speed [rpm]"].astype(int).tolist(),
        "torque": df["Torque [Nm]"].round(1).tolist(),
        "desgaste": df["Tool wear [min]"].astype(int).tolist(),
        "delta_temp": df["delta_temp"].round(2).tolist(),
        "falha": df["Machine failure"].astype(int).tolist(),
        "modos": bits.astype(int).tolist(),
    }


def main() -> None:
    df = carregar()
    dados = {
        "fonte": "AI4I 2020 Predictive Maintenance Dataset, Matzka (2020), UCI ML Repository, DOI 10.24432/C5HS5C",
        "modos": MODOS,
        "limiar_osf": LIMIAR_OSF,
        "regras": validar_regras(df),
        "modelos": treinar(df),
        "registros": registros(df),
    }
    destino = RAIZ / "site" / "data" / "ai4i.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dados, separators=(",", ":")), encoding="utf-8")
    m = dados["modelos"]
    print(f"{len(df)} registros, {int(df['Machine failure'].sum())} falhas")
    for r in dados["regras"]:
        print(f"  regra {r['modo']}: {r['concordancia']:.1%}  FP {r['falsos_positivos']}  FN {r['falsos_negativos']}")
    for modo in MODOS:
        print(f"  F1 {modo}: {m['f1_base'][modo]:.2f} -> {m['f1_eng'][modo]:.2f}")
    print(f"gravado {destino.relative_to(RAIZ)} ({destino.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
