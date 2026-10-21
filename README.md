# energy-forecasting

Hourly weather + electricity-load pipeline on Google Cloud, and a model that forecasts **Connecticut's electricity load
for the next 24 hours, every hour**. Built for Dr. Dave Wanik's courses at UConn (OPIM 5512 Applied Data Science, Lab 5
onward). Every student forks this repo into their **own** energy project on Google Cloud.

## What runs every hour (Eastern time)

| Minute | Cloud Function | Source | Writes to `gs://<bucket>/energy/` |
|---|---|---|---|
| :05 | `scraper-weather-obs` | Iowa State Mesonet, Bradley Airport (BDL) reports | `datasets/bdl_weather_obs_master.csv` (one row per UTC hour) |
| :05 | `scraper-weather-forecast` | [forecast.weather.gov](https://forecast.weather.gov/MapClick.php?lat=41.9389&lon=-72.6832&FcstType=digital), Bradley | `datasets/bdl_weather_forecast_master.csv` (every snapshot: this hour + the next 24, with the NWS issue time) |
| :15 | `scraper-isone` | ISO New England web services, Connecticut zone (4004) | `datasets/ct_load_hourly_master.csv` (official metered, ~2.5 days late), `datasets/ct_load_5min_hourly_master.csv` (live, averaged per hour), `datasets/isone_ct_forecast_master.csv` (ISO-NE's own forecast) |
| :20 | `forecast-24h` | your model in `models/model.joblib` | `forecasts/latest.csv` (24 rows: `target_hour_utc`, `predicted_load_mw`), `forecasts/history/`, `datasets/forecasts_master.csv` |

Raw downloads are kept in `energy/raw/`. All times are stored in **UTC** (`*_utc` columns = the hour the value begins),
so daylight saving never duplicates or drops a row.

## Set up (once)
Follow the **[Energy Project Setup Guide](https://github.com/drdave-teaching/OPIM5512-notebooks/blob/main/Module4/Energy_Project_Setup_F26.ipynb)**:
new Google Cloud project, fork this repo, Workload Identity Federation, 6 repository **variables**
(`PROJECT_ID`, `REGION`, `BUCKET_NAME`, `RUNTIME_SA`, `DEPLOYER_SA`, `WORKLOAD_IDENTITY_PROVIDER`) and 2 repository
**secrets** from your own free ISO Express account (`ISO_NE_USERNAME`, `ISO_NE_PASSWORD`). Without the secrets,
`scraper-isone` deploys but reports `"status": "skipped"`.

Each folder in `cloud_function/` deploys with its own workflow in `.github/workflows/` (on a push to `main` that changes
it, or **Actions → Run workflow**). Get course updates with **Sync fork → Update branch**.

## Train and ship a model
**[Lab 5 notebook](https://github.com/drdave-teaching/OPIM5512-notebooks/blob/main/Module4/Lab5_Energy_Forecast_In_Production.ipynb)**:
Bradley weather + Connecticut load from Jan 2023 to Oct 2026 ([how it was built](https://github.com/drdave-teaching/OPIM5512-notebooks/blob/main/Module4/Lab5_Building_the_Training_Data.ipynb)),
baselines, time-series cross-validation, a tuned model, then upload to `energy/models/model.joblib`.
`forecast-24h` builds its inputs with the same `make_features` function as the notebook, so any scikit-learn model
trained on those 11 features works.

**Just want a working model right now?** In Cloud Shell, from your fork:
```bash
python3 -m venv ~/energy-venv && ~/energy-venv/bin/pip install -q -r starter/requirements.txt
~/energy-venv/bin/python starter/train_and_ship.py --project YOUR-ENERGY-PROJECT-ID --bucket YOUR-BUCKET
gcloud scheduler jobs run forecast-24h-hourly --location=us-central1
```
It trains gradient boosting with the Lab 5 champion settings (about 120 MW average error on 2026, vs 266 MW for "same hour
yesterday") and uploads it; your next 24 hours appear in `energy/forecasts/latest.csv`.

## Data credits
Weather: Iowa State Mesonet (mesonet.agron.iastate.edu) and the National Weather Service. Load and forecasts: ISO New
England (iso-ne.com, ISO Express web services). Each student uses their own ISO Express account.
