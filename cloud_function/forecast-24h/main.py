# forecast-24h: YOUR model forecasts Connecticut electricity load for the next 24 hours, every hour.
#
# Every hour (:20, after the weather scrapers at :05 and the ISO-NE scraper at :15) this function:
#   1. loads your live model:          gs://<bucket>/energy/models/model.joblib (+ model_card.json)
#   2. picks the next 24 FULL hours:   if it runs at 2:20 p.m., it forecasts 3 p.m. today ... 2 p.m. tomorrow
#   3. builds the features with make_features(), the SAME code as the Lab 5 notebook:
#        weather  -> the newest NWS forecast for each hour (bdl_weather_forecast_master.csv)
#        load     -> the load at the same hour yesterday (ct_load_5min_hourly_master.csv, live five-minute load)
#   4. writes 24 rows (target_hour_utc, predicted_load_mw, ...):
#        gs://<bucket>/energy/forecasts/latest.csv                       <- replaced every hour (Lab 6 reads this one)
#        gs://<bucket>/energy/forecasts/history/<issued time>.csv        <- one file per hour, never overwritten
#        gs://<bucket>/energy/datasets/forecasts_master.csv              <- every forecast you ever made, in one table
# No model yet? It politely does nothing ("status": "no_model"). No ISO-NE data yet? "status": "no_load_data".
# HTTP entrypoint: forecast_24h_http

import io
import json
import logging
import os
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
HOURS_AHEAD = int(os.getenv("HOURS_AHEAD", "24"))

MODEL_KEY = f"{ENERGY_PREFIX}/models/model.joblib"
CARD_KEY = f"{ENERGY_PREFIX}/models/model_card.json"
WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
LOAD_KEY = f"{ENERGY_PREFIX}/datasets/ct_load_5min_hourly_master.csv"
LATEST_KEY = f"{ENERGY_PREFIX}/forecasts/latest.csv"
HISTORY_PREFIX = f"{ENERGY_PREFIX}/forecasts/history"
MASTER_KEY = f"{ENERGY_PREFIX}/datasets/forecasts_master.csv"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------------------------------------------------------------------------------------------
# make_features: copied word for word from the Lab 5 notebook (Part 3). Change it there, change it here.
# ----------------------------------------------------------------------------------------------------------------
WEATHER = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct"]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]
FEATURES = CALENDAR + WEATHER + ["load_yesterday_mw"]


def make_features(hours_utc, weather, load):
    """One row of model inputs per hour.
    hours_utc: the hours to describe (UTC).  weather: table indexed by UTC hour with the WEATHER columns.
    load: Connecticut load in MW, indexed by UTC hour (only hours that are already over)."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    local = hours_utc.tz_convert("America/New_York")           # people use electricity on local time
    out = pd.DataFrame(index=hours_utc)
    out["hour"] = local.hour
    out["day_of_week"] = local.dayofweek                       # 0 = Monday ... 6 = Sunday
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(start="2020-01-01", end="2030-12-31")
    out["is_holiday"] = local.tz_localize(None).normalize().isin(holidays).astype(int)

    for column in WEATHER:
        out[column] = weather[column].reindex(hours_utc).values

    # Same hour yesterday. If that hour isn't over yet (only the last hour of a live forecast),
    # use the hour just before it.
    yesterday = load.reindex(hours_utc - pd.Timedelta(hours=24)).values
    hour_before = load.reindex(hours_utc - pd.Timedelta(hours=25)).values
    out["load_yesterday_mw"] = np.where(np.isnan(yesterday), hour_before, yesterday)
    return out[FEATURES]
# ----------------------------------------------------------------------------------------------------------------


def read_csv(bucket, key):
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def latest_weather(forecasts: pd.DataFrame) -> pd.DataFrame:
    """For every target hour, the NEWEST forecast that covers it."""
    forecasts["scraped_hour_utc"] = pd.to_datetime(forecasts["scraped_hour_utc"], utc=True)
    forecasts["target_hour_utc"] = pd.to_datetime(forecasts["target_hour_utc"], utc=True)
    forecasts = forecasts.sort_values(["target_hour_utc", "scraped_hour_utc"])
    newest = forecasts.drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")
    return newest


def run_once(dry_run: bool = False) -> dict:
    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    if not bucket.blob(MODEL_KEY).exists():
        return {"status": "no_model", "message": f"Upload your model to gs://{GCS_BUCKET}/{MODEL_KEY} (Lab 5, Part 7)."}
    model = joblib.load(io.BytesIO(bucket.blob(MODEL_KEY).download_as_bytes()))
    card = {}
    if bucket.blob(CARD_KEY).exists():
        card = json.loads(bucket.blob(CARD_KEY).download_as_text())
    model_version = str(card.get("trained_at", "unknown"))

    forecasts = read_csv(bucket, WEATHER_KEY)
    if forecasts is None or len(forecasts) == 0:
        return {"status": "no_weather_data", "message": "scraper-weather-forecast has not written its master yet."}
    load_table = read_csv(bucket, LOAD_KEY)
    if load_table is None or len(load_table) == 0:
        return {"status": "no_load_data", "message": "No ISO-NE load yet: add the ISO_NE_USERNAME and ISO_NE_PASSWORD "
                "repository secrets, re-run 'Deploy scraper-isone', and wait for its next run (:15)."}
    load_table["hour_utc"] = pd.to_datetime(load_table["hour_utc"], utc=True)
    load = load_table.set_index("hour_utc")["load_mw_5min_avg"].sort_index()

    now = datetime.now(timezone.utc)
    this_hour = pd.Timestamp(now).floor("h")
    targets = pd.date_range(this_hour + pd.Timedelta(hours=1), periods=HOURS_AHEAD, freq="h")
    weather = latest_weather(forecasts)
    X = make_features(targets, weather, load)

    # Honest gap filling (rare): a missing weather value takes its neighbor; a missing yesterday load takes the newest load
    filled_weather = int(X[WEATHER].isna().sum().sum())
    X[WEATHER] = X[WEATHER].interpolate(limit_direction="both")
    filled_load = int(X["load_yesterday_mw"].isna().sum())
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    if X[WEATHER].isna().any().any():
        return {"status": "no_weather_data", "message": "The weather forecast does not cover the next 24 hours yet."}

    preds = model.predict(X[FEATURES])
    out = pd.DataFrame({"issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "target_hour_utc": targets.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lead_hours": np.arange(1, HOURS_AHEAD + 1),
                        "predicted_load_mw": np.round(preds, 1),
                        "model_version": model_version})
    summary = {"status": "ok", "model_version": model_version, "first_target_utc": out["target_hour_utc"].iloc[0],
               "last_target_utc": out["target_hour_utc"].iloc[-1], "newest_load_hour_utc": load.index.max().isoformat(),
               "min_mw": float(out["predicted_load_mw"].min()), "max_mw": float(out["predicted_load_mw"].max()),
               "filled_weather_values": filled_weather, "filled_load_values": filled_load}
    if dry_run:
        summary["dry_run"] = True
        return summary

    csv_text = out.to_csv(index=False)
    bucket.blob(LATEST_KEY).upload_from_string(csv_text, content_type="text/csv")
    bucket.blob(f"{HISTORY_PREFIX}/{now.strftime('%Y%m%dT%H%M%SZ')}.csv").upload_from_string(csv_text, content_type="text/csv")
    old = read_csv(bucket, MASTER_KEY)
    master = out if old is None else pd.concat([old, out], ignore_index=True)
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index=False), content_type="text/csv")
    summary["latest_key"] = LATEST_KEY
    summary["master_rows"] = int(len(master))
    return summary


def forecast_24h_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
