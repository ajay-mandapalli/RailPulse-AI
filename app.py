# ============================================================
# RAILPULSE AI
# Hyderabad MMTS Demand Intelligence System
# Flask Backend
# ============================================================

import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# 1. FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "railpulse-local-development-key"
)


# ============================================================
# 2. FILE PATHS
# ============================================================

DATABASE = "railpulse.db"

DATA_FILE = "data/processed_train.csv"

MODEL_FILE = "models/esn_model.pkl"

X_SCALER_FILE = "models/x_scaler.pkl"

Y_SCALER_FILE = "models/y_scaler.pkl"

MODEL_INFO_FILE = "models/model_info.json"


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

required_files = [
    DATA_FILE,
    MODEL_FILE,
    X_SCALER_FILE,
    Y_SCALER_FILE,
    MODEL_INFO_FILE
]

for file in required_files:

    if not os.path.exists(file):

        raise FileNotFoundError(
            f"Required file not found: {file}"
        )


# ============================================================
# 4. LOAD TRAINED ESN MODEL
# ============================================================

print("\nLoading RailPulse AI model...")

esn_data = joblib.load(
    MODEL_FILE
)

W_in = esn_data["W_in"]

W = esn_data["W"]

LEAK_RATE = esn_data["leak_rate"]

RESERVOIR_SIZE = esn_data["reservoir_size"]

esn_readout = esn_data["readout"]


X_scaler = joblib.load(
    X_SCALER_FILE
)

y_scaler = joblib.load(
    Y_SCALER_FILE
)


with open(
    MODEL_INFO_FILE,
    "r"
) as file:

    model_info = json.load(file)


FEATURES = model_info["features"]

SEQUENCE_LENGTH = model_info["sequence_length"]

SELECTED_MODEL = model_info["selected_model"]


print("Model loaded successfully.")
print("Selected model:", SELECTED_MODEL)


# ============================================================
# 5. LOAD MMTS DATASET
# ============================================================

print("\nLoading MMTS dataset...")

df = pd.read_csv(
    DATA_FILE
)


df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)


df["departure_time"] = (
    df["departure_time"]
    .astype(str)
    .str.strip()
)


df = df.dropna(
    subset=[
        "datetime",
        "from_station",
        "to_station",
        "departure_time"
    ]
).reset_index(drop=True)


print(
    "Dataset loaded:",
    len(df),
    "records"
)


# ============================================================
# 6. DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# 7. CREATE DATABASE TABLES
# ============================================================

def initialize_database():

    connection = get_db_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            full_name TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    # --------------------------------------------------------
    # PREDICTION HISTORY TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            from_station TEXT NOT NULL,

            to_station TEXT NOT NULL,

            prediction_date TEXT NOT NULL,

            departure_time TEXT NOT NULL,

            predicted_passengers INTEGER NOT NULL,

            demand_level TEXT NOT NULL,

            peak_status TEXT NOT NULL,

            model_used TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    # --------------------------------------------------------
    # CREATE ADMIN ACCOUNT
    # --------------------------------------------------------

    admin_username = os.environ.get(
        "ADMIN_USERNAME",
        "admin"
    )

    admin_password = os.environ.get(
        "ADMIN_PASSWORD",
        "admin123"
    )

    admin_full_name = os.environ.get(
        "ADMIN_FULL_NAME",
        "RailPulse Administrator"
    )


    existing_user = cursor.execute(
        """
        SELECT *
        FROM users
        WHERE username = ?
        """,
        (admin_username,)
    ).fetchone()


    if existing_user is None:

        password_hash = generate_password_hash(
            admin_password
        )

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                password,
                full_name
            )
            VALUES (?, ?, ?)
            """,
            (
                admin_username,
                password_hash,
                admin_full_name
            )
        )


    connection.commit()

    connection.close()


# ============================================================
# 8. GET STATION LIST
# ============================================================

def get_stations():

    stations = sorted(
        df["from_station"]
        .dropna()
        .unique()
        .tolist()
    )

    return stations


# ============================================================
# 9. GET DESTINATIONS FOR ORIGIN
# ============================================================

def get_destinations(
    from_station
):

    destinations = (

        df[
            df["from_station"]
            == from_station
        ]["to_station"]

        .dropna()

        .unique()

        .tolist()
    )

    return sorted(destinations)


# ============================================================
# 10. TIME HELPERS
# ============================================================

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


def check_peak_hour(hour):

    return int(
        (7 <= hour < 10)
        or
        (17 <= hour < 20)
    )


# ============================================================
# 11. CREATE ESN RESERVOIR STATE
# ============================================================

def create_esn_state(sequence):

    state = np.zeros(
        RESERVOIR_SIZE,
        dtype=np.float32
    )


    for time_step in sequence:

        input_with_bias = np.concatenate(
            (
                np.array(
                    [1.0],
                    dtype=np.float32
                ),

                time_step.astype(
                    np.float32
                )
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

            (1 - LEAK_RATE)
            * state

            +

            LEAK_RATE
            * new_state
        )


    return state.reshape(
        1,
        -1
    )


# ============================================================
# 12. DEMAND LEVEL
# ============================================================

def get_demand_level(
    passengers
):

    if passengers <= 402:

        return "Low"

    elif passengers <= 642:

        return "Medium"

    else:

        return "High"


# ============================================================
# 13. OPERATIONAL RECOMMENDATION
# ============================================================

def get_recommendation(
    passengers,
    demand_level,
    peak_status
):

    if demand_level == "High":

        return {
            "capacity": (
                "Consider additional coach or service capacity "
                "where operationally feasible."
            ),

            "crowd": (
                "Increase monitoring of platform and boarding "
                "areas because high passenger demand is expected."
            ),

            "planning": (
                "Review nearby high-demand departure slots and "
                "prepare for possible passenger accumulation."
            )
        }


    elif demand_level == "Medium":

        if peak_status == "Peak":

            return {
                "capacity": (
                    "Maintain adequate service capacity for the "
                    "expected moderate demand."
                ),

                "crowd": (
                    "Monitor platform crowding closely because "
                    "this journey occurs during a peak period."
                ),

                "planning": (
                    "Review adjacent departure slots for possible "
                    "increases in passenger demand."
                )
            }

        return {
            "capacity": (
                "Normal planned service capacity should be "
                "sufficient for the expected demand."
            ),

            "crowd": (
                "Continue routine monitoring of passenger flow "
                "at the platform and boarding areas."
            ),

            "planning": (
                "Monitor nearby departure slots for changes in "
                "passenger demand."
            )
        }


    else:

        return {
            "capacity": (
                "Standard service capacity is likely sufficient "
                "for the predicted passenger demand."
            ),

            "crowd": (
                "Routine platform and boarding-area monitoring "
                "should be sufficient."
            ),

            "planning": (
                "No additional demand-management action is "
                "indicated by the current forecast."
            )
        }
# ============================================================
# 14. BUILD SEQUENCE FOR REAL PREDICTION
# ============================================================

def build_prediction_sequence(
    from_station,
    to_station,
    departure_time,
    prediction_date
):

    prediction_datetime = pd.to_datetime(
        prediction_date
    )


    route_history = df[
        (
            df["from_station"]
            == from_station
        )
        &
        (
            df["to_station"]
            == to_station
        )
        &
        (
            df["departure_time"]
            == departure_time
        )
        &
        (
            df["datetime"]
            < prediction_datetime
        )
    ].copy()


    route_history = (
        route_history
        .sort_values("datetime")
        .tail(SEQUENCE_LENGTH)
    )


    if len(route_history) < SEQUENCE_LENGTH:

        return None


    sequence_features = (
        route_history[FEATURES]
        .astype(float)
    )


    scaled_sequence = X_scaler.transform(
        sequence_features
    )


    return scaled_sequence.astype(
        np.float32
    )
# ============================================================
# DAILY CROWD FORECAST
# ============================================================

def generate_daily_crowd_forecast(
    from_station,
    to_station,
    prediction_date
):

    route_rows = df[
        (df["from_station"] == from_station)
        &
        (df["to_station"] == to_station)
    ].copy()

    if route_rows.empty:
        return []

    available_times = (
        route_rows["departure_time"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # Sort departure times correctly
    available_times = sorted(
        available_times,
        key=lambda x: pd.to_datetime(
            x,
            format="%H:%M"
        ).time()
    )

    daily_forecast = []

    for departure_time in available_times:

        sequence = build_prediction_sequence(
            from_station,
            to_station,
            departure_time,
            prediction_date
        )

        if sequence is None:
            continue

        # Create ESN reservoir state
        reservoir_state = create_esn_state(
            sequence
        )

        # ESN prediction
        scaled_prediction = (
            esn_readout.predict(
                reservoir_state
            )
            .reshape(-1, 1)
        )

        passenger_prediction = (
            y_scaler.inverse_transform(
                scaled_prediction
            )[0][0]
        )

        predicted_passengers = max(
            0,
            int(round(passenger_prediction))
        )

        demand_level = get_demand_level(
            predicted_passengers
        )

        daily_forecast.append({
            "time": departure_time,
            "passengers": predicted_passengers,
            "demand": demand_level
        })

    return daily_forecast

# ============================================================
# 15. LOGIN ROUTE
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        connection = get_db_connection()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        connection.close()


        if (
            user
            and
            check_password_hash(
                user["password"],
                password
            )
        ):

            session["user_id"] = user["id"]

            session["username"] = user[
                "username"
            ]

            session["full_name"] = user[
                "full_name"
            ]


            return redirect(
                url_for("dashboard")
            )


        flash(
            "Invalid username or password.",
            "error"
        )


    return render_template(
        "login.html"
    )


# ============================================================
# 16. DASHBOARD
# ============================================================

@app.route(
    "/dashboard"
)
def dashboard():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    connection = get_db_connection()


    recent_predictions = (
        connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()
    )


    total_predictions = (
        connection.execute(
            """
            SELECT COUNT(*)
            AS total
            FROM predictions
            """
        ).fetchone()["total"]
    )


    connection.close()


    average_demand = int(
        round(
            df["passenger_count"].mean()
        )
    )


    return render_template(

        "dashboard.html",

        total_records=len(df),

        total_routes=df[
            "route"
        ].nunique(),

        average_demand=average_demand,

        selected_model=SELECTED_MODEL,

        total_predictions=total_predictions,

        recent_predictions=recent_predictions
    )


# ============================================================
# 17. PREDICTION PAGE
# ============================================================

@app.route(
    "/predict",
    methods=[
        "GET",
        "POST"
    ]
)
def predict():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    stations = get_stations()


    if request.method == "POST":

        from_station = request.form[
            "from_station"
        ]

        to_station = request.form[
            "to_station"
        ]

        prediction_date = request.form[
            "prediction_date"
        ]

        departure_time = request.form[
            "departure_time"
        ]


        # ----------------------------------------------------
        # VERIFY ROUTE
        # ----------------------------------------------------

        route_rows = df[
            (
                df["from_station"]
                == from_station
            )
            &
            (
                df["to_station"]
                == to_station
            )
        ]


        if route_rows.empty:

            flash(
                "The selected MMTS route is not available.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # ----------------------------------------------------
        # VERIFY DEPARTURE TIME
        # ----------------------------------------------------

        available_times = (
            route_rows[
                "departure_time"
            ]
            .astype(str)
            .unique()
            .tolist()
        )


        if departure_time not in available_times:

            flash(
                "The selected departure time is not available "
                "for this route.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # ----------------------------------------------------
        # BUILD HISTORICAL SEQUENCE
        # ----------------------------------------------------

        sequence = build_prediction_sequence(

            from_station,

            to_station,

            departure_time,

            prediction_date
        )


        if sequence is None:

            flash(
                "Not enough historical records are available "
                "before this date. Please select a later date.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # ----------------------------------------------------
        # ESN PREDICTION
        # ----------------------------------------------------

        reservoir_state = (
            create_esn_state(
                sequence
            )
        )


        scaled_prediction = (
            esn_readout.predict(
                reservoir_state
            )
            .reshape(-1, 1)
        )


        passenger_prediction = (
            y_scaler.inverse_transform(
                scaled_prediction
            )[0][0]
        )


        predicted_passengers = max(
            0,
            int(
                round(
                    passenger_prediction
                )
            )
        )


        # ----------------------------------------------------
        # DERIVED INFORMATION
        # ----------------------------------------------------

        selected_datetime = pd.to_datetime(
            f"{prediction_date} {departure_time}"
        ).to_pydatetime()


        hour = selected_datetime.hour


        peak_status = (
            "Peak"
            if check_peak_hour(hour)
            else "Non-Peak"
        )


        demand_level = get_demand_level(
            predicted_passengers
        )


        time_period = get_time_period(
            hour
        )


        recommendation = (
            get_recommendation(
                predicted_passengers,
                demand_level,
                peak_status
            )
        )


        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        connection = get_db_connection()

        cursor = connection.cursor()


        cursor.execute(
            """
            INSERT INTO predictions
            (
                from_station,
                to_station,
                prediction_date,
                departure_time,
                predicted_passengers,
                demand_level,
                peak_status,
                model_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                from_station,
                to_station,
                prediction_date,
                departure_time,
                predicted_passengers,
                demand_level,
                peak_status,
                SELECTED_MODEL
            )
        )


        connection.commit()

        prediction_id = cursor.lastrowid

        connection.close()


        return redirect(
            url_for(
                "prediction_result",
                prediction_id=prediction_id
            )
        )


    return render_template(
        "predict.html",
        stations=stations
    )


# ============================================================
# 18. DESTINATION API
# ============================================================

@app.route(
    "/api/destinations/<from_station>"
)
def destinations_api(
    from_station
):

    destinations = get_destinations(
        from_station
    )

    return jsonify(
        destinations
    )


# ============================================================
# 19. DEPARTURE TIME API
# ============================================================

@app.route(
    "/api/times/<from_station>/<to_station>"
)
def times_api(
    from_station,
    to_station
):

    times = (

        df[
            (
                df["from_station"]
                == from_station
            )
            &
            (
                df["to_station"]
                == to_station
            )
        ]["departure_time"]

        .astype(str)

        .drop_duplicates()

        .tolist()
    )


    return jsonify(
        times
    )


# ============================================================
# 20. RESULT PAGE
# ============================================================

@app.route(
    "/result/<int:prediction_id>"
)
def prediction_result(prediction_id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    connection = get_db_connection()


    prediction = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE id = ?
        """,
        (prediction_id,)
    ).fetchone()


    connection.close()


    if prediction is None:

        return (
            "Prediction not found.",
            404
        )


    recommendation = get_recommendation(
        prediction["predicted_passengers"],
        prediction["demand_level"],
        prediction["peak_status"]
    )


    selected_datetime = pd.to_datetime(
        prediction["prediction_date"]
        + " "
        + prediction["departure_time"]
    ).to_pydatetime()


    time_period = get_time_period(
        selected_datetime.hour
    )
        # --------------------------------------------------------
    # DAILY CROWD FORECAST
    # --------------------------------------------------------

    daily_forecast = generate_daily_crowd_forecast(
        prediction["from_station"],
        prediction["to_station"],
        prediction["prediction_date"]
    )

    if daily_forecast:

        highest_forecast = max(
            daily_forecast,
            key=lambda x: x["passengers"]
        )

        lowest_forecast = min(
            daily_forecast,
            key=lambda x: x["passengers"]
        )

        high_demand_times = [
            item["time"]
            for item in daily_forecast
            if item["demand"] == "High"
        ]

    else:

        highest_forecast = None
        lowest_forecast = None
        high_demand_times = []


    return render_template(
        "result.html",
        prediction=prediction,
        recommendation=recommendation,
        time_period=time_period,
        daily_forecast=daily_forecast,
        highest_forecast=highest_forecast,
        lowest_forecast=lowest_forecast,
        high_demand_times=high_demand_times
    )


# ============================================================
# 21. ANALYTICS
# ============================================================

@app.route(
    "/analytics"
)
def analytics():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    return render_template(
        "analytics.html",
        model_info=model_info
    )


# ============================================================
# 22. PREDICTION HISTORY
# ============================================================

@app.route(
    "/history"
)
def history():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    connection = get_db_connection()


    predictions = (
        connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()
    )


    connection.close()


    return render_template(
        "history.html",
        predictions=predictions
    )


# ============================================================
# 23. LOGOUT
# ============================================================

@app.route(
    "/logout"
)
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# 24. INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# 25. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "RailPulse AI is starting..."
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 60 + "\n")


    app.run(
        debug=True
    )