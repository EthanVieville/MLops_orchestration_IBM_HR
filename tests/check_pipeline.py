"""Vérifie que le pipeline de données est opérationnel."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from mlproject.data import load_data, split
from mlproject.features import build_preprocessor


def check():
    print("=== Chargement des données ===")
    df = load_data()
    print(f"  Shape            : {df.shape}")
    print(f"  Colonnes         : {list(df.columns)}")
    print("  Target (Attrition):")
    for val, count in df["Attrition"].value_counts().items():
        label = "Yes (1)" if val == 1 else "No  (0)"
        pct = count / len(df) * 100
        print(f"    {label} -> {count} ({pct:.1f}%)")

    print("\n=== Split train / test ===")
    X_train, X_test, y_train, y_test = split(df)
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}  (attrition rate: {y_train.mean():.1%})")
    print(f"  y_test  : {y_test.shape}   (attrition rate: {y_test.mean():.1%})")

    print("\n=== Preprocessing ===")
    pre = build_preprocessor()
    X_tr = pre.fit_transform(X_train)
    X_te = pre.transform(X_test)
    print(f"  Features après encoding : {X_tr.shape[1]}")
    print(f"  X_train transformé      : {X_tr.shape}")
    print(f"  X_test  transformé      : {X_te.shape}")

    print("\n[OK] Pipeline opérationnel.")


if __name__ == "__main__":
    check()
