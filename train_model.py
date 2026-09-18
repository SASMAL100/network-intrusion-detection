"""
Trains the Random Forest network intrusion detection model from scratch and
saves the artifacts used by app.py. Reproduces both the misleadingly-optimistic
same-distribution evaluation and the honest evaluation on the official
held-out test set (which contains attack types unseen during training).

Usage:
    python train_model.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def main():
    train_raw = pd.read_csv("data/nsl_kdd_train_original.csv")
    test_raw = pd.read_csv("data/nsl_kdd_test_original.csv")

    df = train_raw.drop("difficulty", axis=1).copy()
    df["label"] = df["attack_type"].apply(lambda x: 0 if x == "normal" else 1)

    le_dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop(["attack_type", "label"], axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    print("== Same-distribution split (misleadingly optimistic) ==")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}  AUC: {roc_auc_score(y_test, proba):.3f}")

    # Official held-out test set - contains attack types unseen in training
    df_official = test_raw.drop("difficulty", axis=1).copy()
    df_official["label"] = df_official["attack_type"].apply(lambda x: 0 if x == "normal" else 1)
    for col in CATEGORICAL_COLS:
        le = le_dict[col]
        df_official[col] = df_official[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df_official[col] = le.transform(df_official[col])

    X_official = df_official.drop(["attack_type", "label"], axis=1)
    y_official = df_official["label"]
    X_official_scaled = scaler.transform(X_official)

    pred_o = model.predict(X_official_scaled)
    proba_o = model.predict_proba(X_official_scaled)[:, 1]
    print("\n== Official held-out test set (unseen attacks, threshold=0.5) ==")
    print(f"Accuracy: {accuracy_score(y_official, pred_o):.3f}  "
          f"Recall: {recall_score(y_official, pred_o):.3f}  "
          f"AUC: {roc_auc_score(y_official, proba_o):.3f}")

    threshold = 0.3
    pred_adj = (proba_o >= threshold).astype(int)
    print(f"\n== Threshold-tuned for recall (threshold={threshold}) ==")
    print(f"Accuracy: {accuracy_score(y_official, pred_adj):.3f}  "
          f"Recall: {recall_score(y_official, pred_adj):.3f}  "
          f"Precision: {precision_score(y_official, pred_adj):.3f}")

    joblib.dump(model, "model/intrusion_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le_dict, "model/label_encoders.pkl")
    joblib.dump(list(X.columns), "model/feature_columns.pkl")

    importances = pd.DataFrame({
        "feature": X.columns, "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv("model/feature_importances.csv", index=False)

    print("\nSaved model, scaler, label_encoders, feature_columns and feature_importances to model/")


if __name__ == "__main__":
    main()
