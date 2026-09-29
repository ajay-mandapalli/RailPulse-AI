# ============================================================
# RAILPULSE AI
# Hyderabad MMTS Passenger Demand Prediction
#
# FINAL MODEL TRAINING
# Models:
#   1. Temporal Convolutional Network (TCN)
#   2. Echo State Network (ESN)
# ============================================================

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tcn import TCN


warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

DATA_FILE = "data/processed_train.csv"

MODEL_DIR = "models"
RESULT_DIR = "static/results"

SEQUENCE_LENGTH = 7

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


print("\n" + "=" * 70)
print("            RAILPULSE AI - FINAL MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        "\nprocessed_train.csv was not found.\n"
        "Run prepare_data.py first."
    )


df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=["datetime", "passenger_count"]
).reset_index(drop=True)


# Sort each route + departure time chronologically
df = df.sort_values(
    [
        "from_station",
        "to_station",
        "departure_time",
        "datetime"
    ]
).reset_index(drop=True)


print("\n[1] Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("Routes:", df["route"].nunique())
print(
    "Date range:",
    df["datetime"].min(),
    "→",
    df["datetime"].max()
)


# ============================================================
# 2. FEATURES
# ============================================================

features = [

    # Route information
    "route_id",
    "distance_km",

    # Time information
    "hour_sin",
    "hour_cos",

    "dow_sin",
    "dow_cos",

    "month_sin",
    "month_cos",

    # Calendar information
    "is_weekend",
    "is_holiday_num",
    "is_peak_hour",

    # Historical demand
    "previous_passenger_count",
    "rolling_avg_3",
    "rolling_avg_5"
]

TARGET = "passenger_count"


missing_columns = [
    col
    for col in features + [TARGET]
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df.dropna(
    subset=features + [TARGET]
).reset_index(drop=True)


print("\n[2] Features verified")

for feature in features:
    print(" •", feature)

print("\nTarget:")
print(" •", TARGET)


# ============================================================
# 3. CHRONOLOGICAL DATE SPLIT
# ============================================================

# IMPORTANT:
# We split using time, not random sampling.

unique_dates = np.array(
    sorted(df["datetime"].dt.date.unique())
)

number_of_dates = len(unique_dates)

train_date_end = int(
    number_of_dates * 0.70
)

val_date_end = int(
    number_of_dates * 0.85
)


train_dates = unique_dates[
    :train_date_end
]

val_dates = unique_dates[
    train_date_end:val_date_end
]

test_dates = unique_dates[
    val_date_end:
]


train_last_date = train_dates[-1]

val_first_date = val_dates[0]
val_last_date = val_dates[-1]

test_first_date = test_dates[0]


print("\n[3] Chronological boundaries created")

print(
    "Training:",
    train_dates[0],
    "→",
    train_last_date
)

print(
    "Validation:",
    val_first_date,
    "→",
    val_last_date
)

print(
    "Testing:",
    test_first_date,
    "→",
    test_dates[-1]
)


# ============================================================
# 4. FIT SCALERS USING TRAINING DATA ONLY
# ============================================================

train_mask = (
    df["datetime"].dt.date <= train_last_date
)

training_rows = df.loc[
    train_mask
].copy()


X_scaler = MinMaxScaler()

y_scaler = MinMaxScaler()


X_scaler.fit(
    training_rows[features]
)

y_scaler.fit(
    training_rows[[TARGET]]
)


print("\n[4] Scalers fitted")
print("IMPORTANT: Scalers used TRAINING DATA ONLY.")


# ============================================================
# 5. SCALE COMPLETE DATA
# ============================================================

scaled_features = X_scaler.transform(
    df[features]
)

scaled_target = y_scaler.transform(
    df[[TARGET]]
)


scaled_features = scaled_features.astype(
    np.float32
)

scaled_target = scaled_target.astype(
    np.float32
)


# Store arrays temporarily in dataframe index order

for i, feature in enumerate(features):

    df[f"_scaled_{feature}"] = (
        scaled_features[:, i]
    )


df["_scaled_target"] = (
    scaled_target[:, 0]
)


# ============================================================
# 6. CREATE ROUTE-WISE TIME SERIES SEQUENCES
# ============================================================

print(
    "\n[5] Creating route-wise "
    f"{SEQUENCE_LENGTH}-step sequences..."
)


X_train_list = []
y_train_list = []

X_val_list = []
y_val_list = []

X_test_list = []
y_test_list = []

test_metadata = []


scaled_feature_columns = [
    f"_scaled_{feature}"
    for feature in features
]


# Each sequence belongs to ONE route and ONE departure slot.

group_columns = [
    "from_station",
    "to_station",
    "departure_time"
]


for group_key, group in df.groupby(
    group_columns,
    sort=False
):

    group = group.sort_values(
        "datetime"
    ).reset_index(drop=True)


    group_X = group[
        scaled_feature_columns
    ].values.astype(np.float32)


    group_y = group[
        "_scaled_target"
    ].values.astype(np.float32)


    # Target begins after 7 historical observations

    for i in range(
        SEQUENCE_LENGTH,
        len(group)
    ):

        # Previous 7 observations
        sequence = group_X[
            i-SEQUENCE_LENGTH:i
        ]

        target_value = group_y[i]

        target_date = (
            group.loc[i, "datetime"].date()
        )


        if target_date <= train_last_date:

            X_train_list.append(
                sequence
            )

            y_train_list.append(
                target_value
            )


        elif target_date <= val_last_date:

            X_val_list.append(
                sequence
            )

            y_val_list.append(
                target_value
            )


        else:

            X_test_list.append(
                sequence
            )

            y_test_list.append(
                target_value
            )

            test_metadata.append({

                "datetime":
                    str(
                        group.loc[
                            i,
                            "datetime"
                        ]
                    ),

                "from_station":
                    group.loc[
                        i,
                        "from_station"
                    ],

                "to_station":
                    group.loc[
                        i,
                        "to_station"
                    ],

                "departure_time":
                    group.loc[
                        i,
                        "departure_time"
                    ],

                "actual_passengers":
                    float(
                        group.loc[
                            i,
                            TARGET
                        ]
                    )
            })


# Convert lists into arrays

X_train_seq = np.asarray(
    X_train_list,
    dtype=np.float32
)

y_train_seq = np.asarray(
    y_train_list,
    dtype=np.float32
).reshape(-1, 1)


X_val_seq = np.asarray(
    X_val_list,
    dtype=np.float32
)

y_val_seq = np.asarray(
    y_val_list,
    dtype=np.float32
).reshape(-1, 1)


X_test_seq = np.asarray(
    X_test_list,
    dtype=np.float32
)

y_test_seq = np.asarray(
    y_test_list,
    dtype=np.float32
).reshape(-1, 1)


# Free large Python lists

del X_train_list
del y_train_list

del X_val_list
del y_val_list

del X_test_list
del y_test_list


print("\nSequence creation completed.")

print(
    "X Train:",
    X_train_seq.shape
)

print(
    "X Validation:",
    X_val_seq.shape
)

print(
    "X Test:",
    X_test_seq.shape
)


print(
    "\nEach TCN input contains:",
    SEQUENCE_LENGTH,
    "historical observations"
)

print(
    "Features per observation:",
    len(features)
)


# ============================================================
# 7. BUILD TCN
# ============================================================

print("\n" + "=" * 70)
print("                   TRAINING TCN")
print("=" * 70)


tcn_model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            len(features)
        )
    ),

    TCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[
            1,
            2,
            4,
            8
        ],
        activation="relu",
        return_sequences=False,
        dropout_rate=0.10
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.15),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])


tcn_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


print("\nTCN Architecture:\n")

tcn_model.summary()


# ============================================================
# 8. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    min_delta=0.00001,

    verbose=1
)


# ============================================================
# 9. TRAIN TCN
# ============================================================

history = tcn_model.fit(

    X_train_seq,
    y_train_seq,

    validation_data=(
        X_val_seq,
        y_val_seq
    ),

    epochs=40,

    batch_size=128,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("\nTCN training completed.")


# ============================================================
# 10. SAVE TCN TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")

plt.title(
    "TCN Training and Validation Loss"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "tcn_loss.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# 11. TCN PREDICTION
# ============================================================

print("\nEvaluating TCN...")


tcn_prediction_scaled = (
    tcn_model.predict(
        X_test_seq,
        batch_size=256,
        verbose=1
    )
)


tcn_predictions = (
    y_scaler.inverse_transform(
        tcn_prediction_scaled
    ).flatten()
)


actual_values = (
    y_scaler.inverse_transform(
        y_test_seq
    ).flatten()
)


# Passenger counts cannot be negative

tcn_predictions = np.maximum(
    tcn_predictions,
    0
)


# ============================================================
# 12. TCN METRICS
# ============================================================

tcn_mae = mean_absolute_error(
    actual_values,
    tcn_predictions
)

tcn_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        tcn_predictions
    )
)

tcn_r2 = r2_score(
    actual_values,
    tcn_predictions
)


print("\n" + "-" * 50)
print("TCN PERFORMANCE")
print("-" * 50)

print(
    "MAE :",
    round(tcn_mae, 2)
)

print(
    "RMSE:",
    round(tcn_rmse, 2)
)

print(
    "R²  :",
    round(tcn_r2, 4)
)


# ============================================================
# 13. TCN ACTUAL VS PREDICTED GRAPH
# ============================================================

plot_count = min(
    250,
    len(actual_values)
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    actual_values[:plot_count],
    label="Actual Demand",
    linewidth=1.5
)

plt.plot(
    tcn_predictions[:plot_count],
    label="TCN Prediction",
    linewidth=1.5
)

plt.xlabel("Test Observation")
plt.ylabel("Passenger Count")

plt.title(
    "TCN - Actual vs Predicted Passenger Demand"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "tcn_actual_vs_predicted.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# 14. SAVE TCN
# ============================================================

tcn_model.save(

    os.path.join(
        MODEL_DIR,
        "tcn_model.keras"
    )
)


print("\nTCN model saved.")


# ============================================================
# 15. ECHO STATE NETWORK
# ============================================================

print("\n" + "=" * 70)
print("                   TRAINING ESN")
print("=" * 70)


# ESN parameters

INPUT_SIZE = len(features)

RESERVOIR_SIZE = 150

SPECTRAL_RADIUS = 0.90

LEAK_RATE = 0.30

INPUT_SCALING = 0.50

RIDGE_ALPHA = 0.001


rng = np.random.default_rng(
    RANDOM_SEED
)


# Input weights

W_in = rng.uniform(

    -INPUT_SCALING,
    INPUT_SCALING,

    size=(
        RESERVOIR_SIZE,
        INPUT_SIZE + 1
    )

).astype(np.float32)


# Reservoir weights

W = rng.uniform(

    -0.5,
    0.5,

    size=(
        RESERVOIR_SIZE,
        RESERVOIR_SIZE
    )

).astype(np.float32)


# ------------------------------------------------------------
# Scale reservoir to desired spectral radius
# ------------------------------------------------------------

eigenvalues = np.linalg.eigvals(W)

current_radius = np.max(
    np.abs(eigenvalues)
)

W = (
    W *
    (
        SPECTRAL_RADIUS /
        current_radius
    )
).astype(np.float32)


print(
    "Reservoir size:",
    RESERVOIR_SIZE
)

print(
    "Spectral radius:",
    SPECTRAL_RADIUS
)

print(
    "Leak rate:",
    LEAK_RATE
)


# ============================================================
# 16. ESN RESERVOIR STATE FUNCTION
# ============================================================

def create_reservoir_states(
    sequences,
    W_in,
    W,
    leak_rate
):

    number_samples = sequences.shape[0]

    states = np.zeros(

        (
            number_samples,
            RESERVOIR_SIZE
        ),

        dtype=np.float32
    )


    for sample_index in range(
        number_samples
    ):

        state = np.zeros(
            RESERVOIR_SIZE,
            dtype=np.float32
        )


        sequence = sequences[
            sample_index
        ]


        for time_step in sequence:

            input_with_bias = np.concatenate(
                (
                    np.array(
                        [1.0],
                        dtype=np.float32
                    ),
                    time_step
                )
            )


            reservoir_input = (
                W_in @ input_with_bias
                +
                W @ state
            )


            new_state = np.tanh(
                reservoir_input
            )


            state = (

                (1 - leak_rate)
                * state

                +

                leak_rate
                * new_state
            )


        states[
            sample_index
        ] = state


    return states


# ============================================================
# 17. GENERATE ESN STATES
# ============================================================

print(
    "\nGenerating ESN training reservoir states..."
)


X_train_esn = create_reservoir_states(

    X_train_seq,
    W_in,
    W,
    LEAK_RATE
)


print(
    "Training states:",
    X_train_esn.shape
)


print(
    "\nGenerating ESN validation reservoir states..."
)


X_val_esn = create_reservoir_states(

    X_val_seq,
    W_in,
    W,
    LEAK_RATE
)


print(
    "Validation states:",
    X_val_esn.shape
)


print(
    "\nGenerating ESN test reservoir states..."
)


X_test_esn = create_reservoir_states(

    X_test_seq,
    W_in,
    W,
    LEAK_RATE
)


print(
    "Test states:",
    X_test_esn.shape
)


# ============================================================
# 18. TRAIN ESN READOUT
# ============================================================

# In ESN, reservoir weights remain fixed.
# Only the output/readout is trained.

esn_readout = Ridge(
    alpha=RIDGE_ALPHA
)


esn_readout.fit(
    X_train_esn,
    y_train_seq.ravel()
)


print("\nESN readout training completed.")


# ============================================================
# 19. ESN PREDICTION
# ============================================================

esn_prediction_scaled = (
    esn_readout.predict(
        X_test_esn
    ).reshape(-1, 1)
)


esn_predictions = (
    y_scaler.inverse_transform(
        esn_prediction_scaled
    ).flatten()
)


esn_predictions = np.maximum(
    esn_predictions,
    0
)


# ============================================================
# 20. ESN METRICS
# ============================================================

esn_mae = mean_absolute_error(
    actual_values,
    esn_predictions
)

esn_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        esn_predictions
    )
)

esn_r2 = r2_score(
    actual_values,
    esn_predictions
)


print("\n" + "-" * 50)
print("ESN PERFORMANCE")
print("-" * 50)

print(
    "MAE :",
    round(esn_mae, 2)
)

print(
    "RMSE:",
    round(esn_rmse, 2)
)

print(
    "R²  :",
    round(esn_r2, 4)
)


# ============================================================
# 21. ESN GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    actual_values[:plot_count],
    label="Actual Demand",
    linewidth=1.5
)

plt.plot(
    esn_predictions[:plot_count],
    label="ESN Prediction",
    linewidth=1.5
)

plt.xlabel("Test Observation")
plt.ylabel("Passenger Count")

plt.title(
    "ESN - Actual vs Predicted Passenger Demand"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "esn_actual_vs_predicted.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# 22. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("                    MODEL COMPARISON")
print("=" * 70)


comparison = pd.DataFrame({

    "Model": [
        "TCN",
        "ESN"
    ],

    "MAE": [
        tcn_mae,
        esn_mae
    ],

    "RMSE": [
        tcn_rmse,
        esn_rmse
    ],

    "R2": [
        tcn_r2,
        esn_r2
    ]
})


print(
    "\n",
    comparison.round(4).to_string(
        index=False
    )
)


# ============================================================
# 23. SELECT WINNING MODEL
# ============================================================

# Primary selection criterion: lowest RMSE
# Secondary information: MAE and R²

if tcn_rmse <= esn_rmse:

    winner = "TCN"

else:

    winner = "ESN"


print("\n" + "*" * 70)

print(
    "DEPLOYED / SELECTED MODEL:",
    winner
)

print("*" * 70)


# ============================================================
# 24. MODEL COMPARISON GRAPH
# ============================================================

models = [
    "TCN",
    "ESN"
]

rmse_values = [
    tcn_rmse,
    esn_rmse
]


plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    models,
    rmse_values
)

plt.ylabel("RMSE")

plt.title(
    "TCN vs ESN - RMSE Comparison"
)

plt.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    rmse_values
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.2f}",

        ha="center",
        va="bottom"
    )


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "model_comparison.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# 25. SAVE ESN
# ============================================================

esn_data = {

    "W_in":
        W_in,

    "W":
        W,

    "leak_rate":
        LEAK_RATE,

    "reservoir_size":
        RESERVOIR_SIZE,

    "readout":
        esn_readout
}


joblib.dump(

    esn_data,

    os.path.join(
        MODEL_DIR,
        "esn_model.pkl"
    )
)


print("\nESN model saved.")


# ============================================================
# 26. SAVE SCALERS
# ============================================================

joblib.dump(

    X_scaler,

    os.path.join(
        MODEL_DIR,
        "x_scaler.pkl"
    )
)


joblib.dump(

    y_scaler,

    os.path.join(
        MODEL_DIR,
        "y_scaler.pkl"
    )
)


print("Scalers saved.")


# ============================================================
# 27. SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "project":
        "RailPulse AI",

    "dataset":
        "Hyderabad MMTS Passenger Demand",

    "selected_model":
        winner,

    "sequence_length":
        SEQUENCE_LENGTH,

    "features":
        features,

    "target":
        TARGET,

    "tcn": {

        "mae":
            float(tcn_mae),

        "rmse":
            float(tcn_rmse),

        "r2":
            float(tcn_r2)
    },

    "esn": {

        "mae":
            float(esn_mae),

        "rmse":
            float(esn_rmse),

        "r2":
            float(esn_r2)
    }
}


with open(

    os.path.join(
        MODEL_DIR,
        "model_info.json"
    ),

    "w"

) as file:

    json.dump(
        model_info,
        file,
        indent=4
    )


# ============================================================
# 28. SAVE TEST PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame(
    test_metadata
)


prediction_results[
    "tcn_prediction"
] = np.round(
    tcn_predictions
).astype(int)


prediction_results[
    "esn_prediction"
] = np.round(
    esn_predictions
).astype(int)


prediction_results.to_csv(

    os.path.join(
        RESULT_DIR,
        "test_predictions.csv"
    ),

    index=False
)


# ============================================================
# 29. SAVE COMPARISON TABLE
# ============================================================

comparison.to_csv(

    os.path.join(
        RESULT_DIR,
        "model_comparison.csv"
    ),

    index=False
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "              RAILPULSE AI - TRAINING COMPLETE"
)

print("=" * 70)


print("\nTCN")

print(
    f"MAE  : {tcn_mae:.2f}"
)

print(
    f"RMSE : {tcn_rmse:.2f}"
)

print(
    f"R²   : {tcn_r2:.4f}"
)


print("\nESN")

print(
    f"MAE  : {esn_mae:.2f}"
)

print(
    f"RMSE : {esn_rmse:.2f}"
)

print(
    f"R²   : {esn_r2:.4f}"
)


print(
    "\nSelected Model:",
    winner
)


print(
    "\nModels saved inside:",
    MODEL_DIR
)

print(
    "Graphs saved inside:",
    RESULT_DIR
)


print("\n" + "=" * 70)