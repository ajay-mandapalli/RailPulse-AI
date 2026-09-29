import os
import pandas as pd
import numpy as np

# ============================================================
# RAILPULSE AI
# Hyderabad MMTS Passenger Demand Prediction
# Stage 1: Data Preparation & Feature Engineering
# ============================================================

INPUT_FILE = "data/train.csv"
OUTPUT_FILE = "data/processed_train.csv"


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("       RAILPULSE AI - DATA PREPARATION")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found: {INPUT_FILE}\n"
        "Make sure train.csv is inside the data folder."
    )

df = pd.read_csv(INPUT_FILE)

print("\n[1] Dataset loaded successfully")
print("Original shape:", df.shape)


# ------------------------------------------------------------
# 2. REMOVE EXACT DUPLICATES
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print("\n[2] Duplicate check completed")
print("Duplicates found:", duplicate_count)


# ------------------------------------------------------------
# 3. CONVERT DATE
# ------------------------------------------------------------

# Dataset date format: DD-MM-YYYY
df["service_date"] = pd.to_datetime(
    df["service_date"],
    format="%d-%m-%Y",
    errors="coerce"
)


# ------------------------------------------------------------
# 4. CLEAN DEPARTURE TIME
# ------------------------------------------------------------

df["departure_time"] = (
    df["departure_time"]
    .astype(str)
    .str.strip()
)

df["datetime"] = pd.to_datetime(
    df["service_date"].dt.strftime("%Y-%m-%d")
    + " "
    + df["departure_time"],
    errors="coerce"
)


# ------------------------------------------------------------
# 5. REMOVE INVALID ESSENTIAL ROWS
# ------------------------------------------------------------

before_cleaning = len(df)

df = df.dropna(
    subset=[
        "service_date",
        "datetime",
        "from_station",
        "to_station",
        "distance_km",
        "passenger_count"
    ]
).reset_index(drop=True)

removed_rows = before_cleaning - len(df)

print("\n[3] Date/time cleaning completed")
print("Invalid rows removed:", removed_rows)


# ------------------------------------------------------------
# 6. SORT CHRONOLOGICALLY
# ------------------------------------------------------------

df = df.sort_values(
    ["datetime", "from_station", "to_station"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 7. CREATE ROUTE INFORMATION
# ------------------------------------------------------------

df["route"] = (
    df["from_station"].astype(str)
    + " → "
    + df["to_station"].astype(str)
)

# Stable numeric route ID
route_names = sorted(df["route"].unique())

route_mapping = {
    route: index
    for index, route in enumerate(route_names)
}

df["route_id"] = df["route"].map(route_mapping)


# ------------------------------------------------------------
# 8. TIME FEATURES
# ------------------------------------------------------------

df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

# Monday = 0, Sunday = 6
df["day_of_week_num"] = df["datetime"].dt.dayofweek

df["day_name"] = df["datetime"].dt.day_name()

df["is_weekend"] = (
    df["day_of_week_num"] >= 5
).astype(int)

df["month_num"] = df["datetime"].dt.month
df["month_name"] = df["datetime"].dt.month_name()

df["quarter"] = df["datetime"].dt.quarter


# ------------------------------------------------------------
# 9. PEAK-HOUR FEATURE
# ------------------------------------------------------------

# Morning peak: 07:00–10:00
# Evening peak: 17:00–20:00

df["is_peak_hour"] = (
    ((df["hour"] >= 7) & (df["hour"] < 10))
    |
    ((df["hour"] >= 17) & (df["hour"] < 20))
).astype(int)


# ------------------------------------------------------------
# 10. TIME PERIOD
# ------------------------------------------------------------

def get_time_period(hour):

    if 5 <= hour < 7:
        return "Early Morning"

    elif 7 <= hour < 10:
        return "Morning Peak"

    elif 10 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 16:
        return "Afternoon"

    elif 16 <= hour < 20:
        return "Evening Peak"

    elif 20 <= hour < 23:
        return "Evening"

    else:
        return "Night"


df["time_period"] = df["hour"].apply(get_time_period)


# ------------------------------------------------------------
# 11. SEASON
# ------------------------------------------------------------

def get_season(month):

    if month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    elif month in [10, 11]:
        return "Post-Monsoon"

    else:
        return "Winter"


df["season"] = df["month_num"].apply(get_season)


# ------------------------------------------------------------
# 12. HOLIDAY ENCODING
# ------------------------------------------------------------

def convert_holiday(value):

    value = str(value).strip().lower()

    if value in ["true", "1", "yes"]:
        return 1

    return 0


df["is_holiday_num"] = df["is_holiday"].apply(convert_holiday)


# ------------------------------------------------------------
# 13. CYCLICAL TIME FEATURES
# ------------------------------------------------------------

# These help ML models understand that:
# 23:00 and 00:00 are close together,
# December and January are also close together.

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week_num"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week_num"] / 7
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month_num"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month_num"] / 12
)


# ------------------------------------------------------------
# 14. HISTORICAL ROUTE DEMAND FEATURES
# ------------------------------------------------------------

# IMPORTANT:
# shift(1) ensures the current passenger_count is NOT used
# to predict itself.

# Group by route AND departure time.
# This means historical demand is compared with the same
# train route at the same departure slot.

route_groups = df.groupby(
    ["from_station", "to_station", "departure_time"],
    sort=False
)

df["previous_passenger_count"] = (
    route_groups["passenger_count"]
    .shift(1)
)

df["rolling_avg_3"] = (
    route_groups["passenger_count"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

df["rolling_avg_5"] = (
    route_groups["passenger_count"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)


# ------------------------------------------------------------
# 15. HANDLE INITIAL LAG VALUES
# ------------------------------------------------------------

# First observation of each route has no historical value.
# Keep these rows but mark missing history using route-level
# historical averages calculated only from earlier observations.

df["previous_passenger_count"] = (
    df["previous_passenger_count"]
    .fillna(0)
)

df["rolling_avg_3"] = (
    df["rolling_avg_3"]
    .fillna(0)
)

df["rolling_avg_5"] = (
    df["rolling_avg_5"]
    .fillna(0)
)


# ------------------------------------------------------------
# 16. DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n[4] Feature engineering completed")

print("\nRoutes:", df["route"].nunique())
print("Stations:", df["from_station"].nunique())

print(
    "Date range:",
    df["service_date"].min().date(),
    "to",
    df["service_date"].max().date()
)

print(
    "Passenger range:",
    int(df["passenger_count"].min()),
    "to",
    int(df["passenger_count"].max())
)


# ------------------------------------------------------------
# 17. SAVE PROCESSED DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    date_format="%Y-%m-%d"
)

print("\n[5] Processed dataset saved successfully")
print("File:", OUTPUT_FILE)
print("Final shape:", df.shape)


# ------------------------------------------------------------
# 18. DISPLAY CREATED FEATURES
# ------------------------------------------------------------

created_features = [
    "route",
    "route_id",
    "datetime",
    "hour",
    "minute",
    "day_of_week_num",
    "day_name",
    "is_weekend",
    "month_num",
    "month_name",
    "quarter",
    "is_peak_hour",
    "time_period",
    "season",
    "is_holiday_num",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "previous_passenger_count",
    "rolling_avg_3",
    "rolling_avg_5"
]

print("\nNew / engineered features:")
for feature in created_features:
    print(" •", feature)

print("\n" + "=" * 60)
print("       DATA PREPARATION COMPLETED")
print("=" * 60)