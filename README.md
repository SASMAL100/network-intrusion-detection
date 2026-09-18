# NetSentry — Network Intrusion Detection

A security-operations-console style demo that flags anomalous network
connections in real time, trained on the NSL-KDD benchmark dataset. Built
with scikit-learn and Streamlit.

**Live demo:** https://network-intrusion-detection-fast.streamlit.app

## What it does

- Classifies a network connection as normal or an intrusion attempt from
  41 connection-level features (protocol, service, byte counts, error
  rates, etc.) using a Random Forest classifier.
- Live "Detection Log" console — every scan run in a session is appended
  to a running log, like a real SOC monitoring console.
- An adjustable detection sensitivity (threshold) slider, since in
  security contexts missing an attack is costlier than an extra false alarm.
- A dataset explorer (original NSL-KDD data included, downloadable) and a
  model internals tab that reports performance honestly.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — a refined,
duplicate-free version of the classic KDD Cup 1999 intrusion detection
benchmark. 125,973 labeled training connections, 41 features, spanning
normal traffic and 22 attack types (DoS, probe, R2L, U2R). The official
train/test split is included in this repo:
`data/nsl_kdd_train_original.csv` and `data/nsl_kdd_test_original.csv`.

## Model performance — reported honestly

A same-distribution train/test split on this dataset is a well-known trap:
attack signatures repeat between train and test, producing misleadingly
high scores. This project reports both:

| Evaluation | Accuracy | Recall | AUC |
|---|---|---|---|
| Same-distribution split | 99.9% | – | 1.000 |
| **Official test set (unseen attacks), threshold=0.5** | 77.7% | 63.0% | 0.962 |
| **Official test set, threshold tuned to 0.3** | 82.5% | 71.6% | 0.962 |

The official NSL-KDD test set contains attack types that never appear in
training, so it's a genuine test of generalization rather than
memorization. Lowering the classification threshold trades a little
precision for meaningfully better recall — the right trade-off for a
security context, since a missed attack is more costly than an extra
alert.

The strongest predictors (by feature importance) are `src_bytes`,
`same_srv_rate`, and connection error rates — consistent with how DoS and
probing attacks actually look on the wire.

## Project structure

```
├── app.py                                # Streamlit app (Live Scan / Traffic Data / Model Internals)
├── train_model.py                        # Reproducible training script
├── data/
│   ├── nsl_kdd_train_original.csv        # Official NSL-KDD training set, as sourced
│   └── nsl_kdd_test_original.csv         # Official NSL-KDD test set (unseen attacks)
├── model/
│   ├── intrusion_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.pkl
│   └── feature_importances.csv
├── .streamlit/config.toml                # App theme
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain from scratch:

```bash
python train_model.py
```

## Limitations

This is a portfolio/demo project built on a research benchmark dataset,
not a production intrusion detection system. NSL-KDD traffic patterns are
now dated; a real deployment would need continuous retraining against
live traffic, since attack patterns evolve.

## Tech stack

Python, scikit-learn, pandas, Streamlit, Plotly
