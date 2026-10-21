"""Train a starter forecasting model and ship it to your bucket, in one command (run it in Cloud Shell).

It does what Lab 5 Parts 3-7 do: same data, same features (make_features is imported from forecast-24h, so production
and training can never disagree), gradient boosting with the Lab 5 champion settings, then uploads
gs://YOUR-BUCKET/energy/models/model.joblib + model_card.json. forecast-24h picks it up at its next run (:20).

Cloud Shell:
    git clone https://github.com/YOUR-GITHUB-USER/YOUR-ENERGY-FORK.git && cd YOUR-ENERGY-FORK
    python3 -m venv ~/energy-venv && ~/energy-venv/bin/pip install -q -r starter/requirements.txt
    ~/energy-venv/bin/python starter/train_and_ship.py --project YOUR-ENERGY-PROJECT-ID --bucket YOUR-BUCKET
    gcloud scheduler jobs run forecast-24h-hourly --location=us-central1      # forecast right now instead of at :20

Add --dry-run to train and score without uploading anything.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cloud_function", "forecast-24h"))
from main import FEATURES, make_features   # noqa: E402  (the exact production feature code)

DATA_URL = "https://raw.githubusercontent.com/drdave-teaching/OPIM5512-notebooks/main/Module4/data/bdl_ct_load_hourly_2023_2026.csv"
PARAMS = {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31, "random_state": 42}   # Lab 5 champion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="")
    parser.add_argument("--bucket", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and (args.project == "" or args.bucket == ""):
        sys.exit("Give --project and --bucket (your energy project ID, usually the same for both), or use --dry-run.")

    print("1/4 Loading the Lab 5 training data from GitHub ...")
    df = pd.read_csv(DATA_URL, index_col="hour_utc", parse_dates=["hour_utc"])
    X_all = make_features(df.index, df, df["load_mw"])
    keep = X_all["load_yesterday_mw"].notna()
    X_all = X_all[keep]
    y_all = df.loc[keep, "load_mw"]
    years = X_all.index.tz_convert("America/New_York").year
    print("    ", len(X_all), "hours,", df["hour_local"].iloc[0], "to", df["hour_local"].iloc[-1])

    print("2/4 Honest check: train on 2023-2025, test on 2026 so far ...")
    check = HistGradientBoostingRegressor(**PARAMS)
    check.fit(X_all[years <= 2025], y_all[years <= 2025])
    test_mae = mean_absolute_error(y_all[years == 2026], check.predict(X_all[years == 2026]))
    baseline_mae = mean_absolute_error(y_all[years == 2026], X_all.loc[years == 2026, "load_yesterday_mw"])
    print("     test MAE", round(test_mae, 1), "MW | same-hour-yesterday baseline", round(baseline_mae, 1), "MW")

    print("3/4 Final model: train on everything (2023 to the last day of data) ...")
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(X_all, y_all)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    card = {"model_type": type(model).__name__, "champion": "gradient_boosting (starter, train_and_ship.py)",
            "best_params": {"learning_rate": "0.05", "max_iter": "600", "max_leaf_nodes": "31"},
            "trained_at": trained_at, "sklearn_version": sklearn.__version__, "features": FEATURES,
            "test_mae_mw_2026": round(float(test_mae), 1), "baseline_same_hour_yesterday_mae_mw_2026": round(float(baseline_mae), 1),
            "train_period": str(X_all.index.min()) + " to " + str(X_all.index.max()), "n_train": int(len(X_all))}
    joblib.dump(model, "/tmp/model.joblib", compress=3)
    with open("/tmp/model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print("     saved /tmp/model.joblib (" + str(round(os.path.getsize("/tmp/model.joblib") / 1e6, 1)) + " MB), scikit-learn", sklearn.__version__)

    if args.dry_run:
        print("4/4 --dry-run: nothing uploaded.")
        return
    from google.cloud import storage
    print("4/4 Uploading to gs://" + args.bucket + "/energy/models/ ...")
    bucket = storage.Client(project=args.project).bucket(args.bucket)
    stamp = trained_at.replace(":", "-")
    bucket.blob("energy/models/model.joblib").upload_from_filename("/tmp/model.joblib")
    bucket.blob("energy/models/model_card.json").upload_from_filename("/tmp/model_card.json")
    bucket.blob("energy/models/history/model_" + stamp + ".joblib").upload_from_filename("/tmp/model.joblib")
    bucket.blob("energy/models/history/model_card_" + stamp + ".json").upload_from_filename("/tmp/model_card.json")
    print("     done. forecast-24h uses it at its next run (:20), or right now:")
    print("     gcloud scheduler jobs run forecast-24h-hourly --location=us-central1")
    print("     then: gcloud storage cat gs://" + args.bucket + "/energy/forecasts/latest.csv")


if __name__ == "__main__":
    main()
