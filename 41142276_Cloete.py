"""
BWIB621 / STTN627 Practical Project
Bike-sharing demand modelling

Models:
A1. Additive GAM
A2. GAM with temperature x windspeed interaction
B1. Random forest
B2. XGBoost
C1. Linear AR(7)
C2. Feedforward neural network
C3. Simple RNN
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from pygam import LinearGAM, s, f, te

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


warnings.filterwarnings("ignore")


# ================================================================
# REPRODUCIBILITY
# ================================================================

RANDOM_SEED = 2026

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

RESULTS = pd.DataFrame(
    columns=["part", "model", "target", "metric", "value"]
)

tscv = TimeSeriesSplit(n_splits=5)


# ================================================================
# FEATURES
# ================================================================

FEATURES = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def add_result(part, model, target, metric, value):
    RESULTS.loc[len(RESULTS)] = [
        part,
        model,
        target,
        metric,
        float(value)
    ]


def calculate_rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )


def save_current_plot(filename):
    plt.savefig(
        os.path.join(FIG_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# ================================================================
# DATA LOADING
# ================================================================

print("\n================================================")
print("LOADING DATA")
print("================================================")

from ucimlrepo import fetch_ucirepo

bike_sharing = fetch_ucirepo(id=275)

X = bike_sharing.data.features.copy()
y = bike_sharing.data.targets.copy()

df = pd.concat([X, y], axis=1)


# If hourly data are returned, aggregate to daily data.
if "hr" in df.columns:

    print("Hourly data detected.")
    print("Aggregating hourly observations to daily observations.")

    aggregation_rules = {
        "season": "first",
        "yr": "first",
        "mnth": "first",
        "holiday": "first",
        "weekday": "first",
        "workingday": "first",
        "weathersit": "max",
        "temp": "mean",
        "atemp": "mean",
        "hum": "mean",
        "windspeed": "mean",
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    }

    aggregation_rules = {
        column: method
        for column, method in aggregation_rules.items()
        if column in df.columns
    }

    df = (
        df.groupby("dteday")
        .agg(aggregation_rules)
        .reset_index()
    )

else:

    print("Daily data detected.")


df["dteday"] = pd.to_datetime(df["dteday"])

df = (
    df.sort_values("dteday")
    .reset_index(drop=True)
)


required_columns = [
    "dteday",
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


print("\nDataset dimensions:")
print(df.shape)

print("\nDate range:")
print(
    df["dteday"].min().date(),
    "to",
    df["dteday"].max().date()
)


# ================================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ================================================================

print("\n================================================")
print("CHRONOLOGICAL TRAIN/TEST SPLIT")
print("================================================")

cutoff_date = "2012-09-01"
cutoff = pd.Timestamp(cutoff_date)

train_df = df[
    df["dteday"] < cutoff
].copy()

test_df = df[
    df["dteday"] >= cutoff
].copy()

print(
    f"Training observations: {len(train_df)} "
    f"({train_df['dteday'].min().date()} to "
    f"{train_df['dteday'].max().date()})"
)

print(
    f"Testing observations: {len(test_df)} "
    f"({test_df['dteday'].min().date()} to "
    f"{test_df['dteday'].max().date()})"
)


# Day of year.
train_df["doy"] = (
    train_df["dteday"].dt.dayofyear
)

test_df["doy"] = (
    test_df["dteday"].dt.dayofyear
)


# ================================================================
# PART A1: ADDITIVE GAM
# ================================================================

print("\n================================================")
print("PART A1: ADDITIVE GAM")
print("================================================")

gam_columns = [
    "temp",
    "hum",
    "windspeed",
    "doy",
    "season",
    "weekday"
]

X_train_gam = train_df[
    gam_columns
].values

X_test_gam = test_df[
    gam_columns
].values

y_train_gam = train_df["cnt"].values
y_test_gam = test_df["cnt"].values


gam = LinearGAM(
    s(0)
    + s(1)
    + s(2)
    + s(3, basis="cp")
    + f(4)
    + f(5)
)

print("Fitting additive GAM...")

gam.gridsearch(
    X_train_gam,
    y_train_gam,
    progress=False
)

gam_predictions = gam.predict(
    X_test_gam
)

gam_rmse = calculate_rmse(
    y_test_gam,
    gam_predictions
)

add_result(
    "A1",
    "GAM additive",
    "cnt",
    "RMSE",
    gam_rmse
)

print(
    f"Additive GAM test RMSE: "
    f"{gam_rmse:.3f}"
)


# ------------------------------------------------
# FIXED GAM PARTIAL-EFFECT PLOTS
# ------------------------------------------------

print("Creating GAM partial-effect plots...")

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 9)
)

axes = axes.flatten()


# Valid reference values for every variable.
# This prevents pyGAM from using zero for season.
reference_values = np.array(
    [
        train_df["temp"].median(),
        train_df["hum"].median(),
        train_df["windspeed"].median(),
        train_df["doy"].median(),
        train_df["season"].mode()[0],
        train_df["weekday"].mode()[0]
    ],
    dtype=float
)


for term_number, column_name in enumerate(gam_columns):

    # Create a valid base grid.
    grid = np.tile(
        reference_values,
        (100, 1)
    )

    # Continuous predictors.
    if column_name == "temp":

        grid[:, term_number] = np.linspace(
            train_df["temp"].min(),
            train_df["temp"].max(),
            100
        )

    elif column_name == "hum":

        grid[:, term_number] = np.linspace(
            train_df["hum"].min(),
            train_df["hum"].max(),
            100
        )

    elif column_name == "windspeed":

        grid[:, term_number] = np.linspace(
            train_df["windspeed"].min(),
            train_df["windspeed"].max(),
            100
        )

    elif column_name == "doy":

        grid[:, term_number] = np.linspace(
            1,
            366,
            100
        )

    # Categorical predictors.
    elif column_name == "season":

        grid = np.tile(
            reference_values,
            (4, 1)
        )

        grid[:, term_number] = np.array(
            [1, 2, 3, 4],
            dtype=float
        )

    elif column_name == "weekday":

        grid = np.tile(
            reference_values,
            (7, 1)
        )

        grid[:, term_number] = np.array(
            [0, 1, 2, 3, 4, 5, 6],
            dtype=float
        )

    partial_effect, confidence_interval = (
        gam.partial_dependence(
            term=term_number,
            X=grid,
            width=0.95
        )
    )

    if column_name in ["season", "weekday"]:

        lower_errors = (
            partial_effect
            - confidence_interval[:, 0]
        )

        upper_errors = (
            confidence_interval[:, 1]
            - partial_effect
        )

        axes[term_number].errorbar(
            grid[:, term_number],
            partial_effect,
            yerr=[
                lower_errors,
                upper_errors
            ],
            fmt="o-",
            color="navy",
            capsize=4,
            linewidth=2
        )

        if column_name == "season":

            axes[term_number].set_xticks(
                [1, 2, 3, 4]
            )

        else:

            axes[term_number].set_xticks(
                [0, 1, 2, 3, 4, 5, 6]
            )

    else:

        axes[term_number].plot(
            grid[:, term_number],
            partial_effect,
            color="navy",
            linewidth=2
        )

        axes[term_number].fill_between(
            grid[:, term_number],
            confidence_interval[:, 0],
            confidence_interval[:, 1],
            color="lightblue",
            alpha=0.5
        )

    axes[term_number].set_title(
        "GAM partial effect: "
        + column_name
    )

    axes[term_number].set_xlabel(
        column_name
    )

    axes[term_number].set_ylabel(
        "Partial effect"
    )

    axes[term_number].grid(
        alpha=0.3
    )


plt.suptitle(
    "Partial effects from additive GAM",
    fontsize=16
)

plt.tight_layout()

save_current_plot(
    "A1_GAM_partial_effects.png"
)


# ------------------------------------------------
# ORDINARY LINEAR REGRESSION
# ------------------------------------------------

print("\nFitting ordinary linear regression...")

linear_model = LinearRegression()

linear_model.fit(
    X_train_gam,
    y_train_gam
)

linear_predictions = linear_model.predict(
    X_test_gam
)

linear_rmse = calculate_rmse(
    y_test_gam,
    linear_predictions
)

add_result(
    "A1",
    "Linear regression",
    "cnt",
    "RMSE",
    linear_rmse
)

print(
    f"Ordinary linear regression test RMSE: "
    f"{linear_rmse:.3f}"
)


# ================================================================
# PART A2: GAM INTERACTION
# ================================================================

print("\n================================================")
print("PART A2: GAM INTERACTION")
print("================================================")

interaction_columns = [
    "temp",
    "windspeed",
    "hum",
    "doy",
    "season",
    "weekday"
]

X_train_interaction = train_df[
    interaction_columns
].values

X_test_interaction = test_df[
    interaction_columns
].values


gam_interaction = LinearGAM(
    te(0, 1)
    + s(2)
    + s(3, basis="cp")
    + f(4)
    + f(5)
)

print("Fitting interaction GAM...")

gam_interaction.gridsearch(
    X_train_interaction,
    y_train_gam,
    progress=False
)

interaction_predictions = (
    gam_interaction.predict(
        X_test_interaction
    )
)

interaction_rmse = calculate_rmse(
    y_test_gam,
    interaction_predictions
)

add_result(
    "A2",
    "GAM temperature-windspeed interaction",
    "cnt",
    "RMSE",
    interaction_rmse
)

print(
    f"Interaction GAM test RMSE: "
    f"{interaction_rmse:.3f}"
)


# ------------------------------------------------
# INTERACTION SURFACE
# ------------------------------------------------

temperature_values = np.linspace(
    train_df["temp"].min(),
    train_df["temp"].max(),
    60
)

windspeed_values = np.linspace(
    train_df["windspeed"].min(),
    train_df["windspeed"].max(),
    60
)

temperature_grid, windspeed_grid = np.meshgrid(
    temperature_values,
    windspeed_values
)


surface_input = np.column_stack(
    [
        temperature_grid.ravel(),
        windspeed_grid.ravel(),
        np.full(
            temperature_grid.size,
            train_df["hum"].median()
        ),
        np.full(
            temperature_grid.size,
            train_df["doy"].median()
        ),
        np.full(
            temperature_grid.size,
            train_df["season"].mode()[0]
        ),
        np.full(
            temperature_grid.size,
            train_df["weekday"].mode()[0]
        )
    ]
)

surface_predictions = gam_interaction.predict(
    surface_input
)

surface_predictions = surface_predictions.reshape(
    temperature_grid.shape
)


plt.figure(figsize=(10, 7))

contour = plt.contourf(
    temperature_grid,
    windspeed_grid,
    surface_predictions,
    levels=25,
    cmap="viridis"
)

plt.colorbar(
    contour,
    label="Predicted daily rentals"
)

plt.xlabel("Temperature")
plt.ylabel("Windspeed")

plt.title(
    "GAM interaction surface: temperature x windspeed"
)

save_current_plot(
    "A2_GAM_interaction_surface.png"
)


# ================================================================
# PART B1: RANDOM FOREST
# ================================================================

print("\n================================================")
print("PART B1: RANDOM FOREST")
print("================================================")

X_train_tree = train_df[
    FEATURES
].values

X_test_tree = test_df[
    FEATURES
].values

y_train_tree = train_df["cnt"].values
y_test_tree = test_df["cnt"].values


random_forest = RandomForestRegressor(
    n_estimators=500,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

random_forest_grid = {
    "max_features": [
        "sqrt",
        0.5,
        0.75,
        1.0
    ]
}

random_forest_search = GridSearchCV(
    estimator=random_forest,
    param_grid=random_forest_grid,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    n_jobs=-1,
    verbose=0
)

print("Tuning random forest...")

random_forest_search.fit(
    X_train_tree,
    y_train_tree
)

best_random_forest = (
    random_forest_search.best_estimator_
)

random_forest_predictions = (
    best_random_forest.predict(
        X_test_tree
    )
)

random_forest_rmse = calculate_rmse(
    y_test_tree,
    random_forest_predictions
)

add_result(
    "B1",
    "Random forest",
    "cnt",
    "RMSE",
    random_forest_rmse
)

print(
    "Best random forest parameters:",
    random_forest_search.best_params_
)

print(
    f"Random forest test RMSE: "
    f"{random_forest_rmse:.3f}"
)


# ------------------------------------------------
# RANDOM FOREST IMPORTANCE
# ------------------------------------------------

random_forest_importance = pd.Series(
    best_random_forest.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=True
)


plt.figure(figsize=(10, 7))

random_forest_importance.plot(
    kind="barh",
    color="forestgreen"
)

plt.xlabel("Feature importance")
plt.ylabel("Predictor")
plt.title("Random forest variable importance")
plt.grid(
    axis="x",
    alpha=0.3
)

save_current_plot(
    "B1_random_forest_importance.png"
)


# ================================================================
# PART B2: XGBOOST
# ================================================================

print("\n================================================")
print("PART B2: XGBOOST")
print("================================================")

fixed_depth = 3

xgb_base_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=fixed_depth,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric="rmse"
)

xgb_grid = {
    "n_estimators": [
        100,
        300,
        500,
        800
    ],
    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.10
    ]
}

xgb_search = GridSearchCV(
    estimator=xgb_base_model,
    param_grid=xgb_grid,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    n_jobs=-1,
    verbose=0
)

print("Tuning XGBoost...")

xgb_search.fit(
    X_train_tree,
    y_train_tree
)

best_xgb_parameters = (
    xgb_search.best_params_
)

print(
    "Best XGBoost parameters:",
    best_xgb_parameters
)


best_xgb = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=fixed_depth,
    n_estimators=best_xgb_parameters[
        "n_estimators"
    ],
    learning_rate=best_xgb_parameters[
        "learning_rate"
    ],
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric="rmse"
)

best_xgb.fit(
    X_train_tree,
    y_train_tree,
    eval_set=[
        (X_train_tree, y_train_tree),
        (X_test_tree, y_test_tree)
    ],
    verbose=False
)

xgb_predictions = best_xgb.predict(
    X_test_tree
)

xgb_rmse = calculate_rmse(
    y_test_tree,
    xgb_predictions
)

add_result(
    "B2",
    "XGBoost",
    "cnt",
    "RMSE",
    xgb_rmse
)

print(
    f"XGBoost test RMSE: "
    f"{xgb_rmse:.3f}"
)


# ------------------------------------------------
# XGBOOST LEARNING CURVE
# ------------------------------------------------

xgb_evaluation_results = (
    best_xgb.evals_result()
)

xgb_train_rmse_curve = (
    xgb_evaluation_results[
        "validation_0"
    ]["rmse"]
)

xgb_test_rmse_curve = (
    xgb_evaluation_results[
        "validation_1"
    ]["rmse"]
)

plt.figure(figsize=(10, 6))

plt.plot(
    range(
        1,
        len(xgb_train_rmse_curve) + 1
    ),
    xgb_train_rmse_curve,
    label="Training RMSE"
)

plt.plot(
    range(
        1,
        len(xgb_test_rmse_curve) + 1
    ),
    xgb_test_rmse_curve,
    label="Test RMSE"
)

plt.xlabel("Boosting iteration")
plt.ylabel("RMSE")
plt.title("XGBoost training and test error")
plt.legend()
plt.grid(alpha=0.3)

save_current_plot(
    "B2_XGBoost_learning_curve.png"
)


# ================================================================
# PART C: BUILD LAGGED DATA
# ================================================================

print("\n================================================")
print("PART C: SEQUENCE MODELS")
print("================================================")

lagged_df = (
    df.sort_values("dteday")
    .reset_index(drop=True)
    .copy()
)

lag_columns = []


for lag in range(1, 8):

    column_name = f"cnt_lag{lag}"

    lagged_df[column_name] = (
        lagged_df["cnt"].shift(lag)
    )

    lag_columns.append(column_name)


for predictor in ["temp", "windspeed"]:

    for lag in range(1, 8):

        column_name = f"{predictor}_lag{lag}"

        lagged_df[column_name] = (
            lagged_df[predictor].shift(lag)
        )

        lag_columns.append(column_name)


lagged_df = (
    lagged_df.dropna()
    .reset_index(drop=True)
)


train_lagged = lagged_df[
    lagged_df["dteday"] < cutoff
].copy()

test_lagged = lagged_df[
    lagged_df["dteday"] >= cutoff
].copy()


X_train_flat = train_lagged[
    lag_columns
].values

X_test_flat = test_lagged[
    lag_columns
].values

y_train_sequence = train_lagged[
    "cnt"
].values

y_test_sequence = test_lagged[
    "cnt"
].values


# ================================================================
# SCALE FLAT SEQUENCE DATA
# ================================================================

input_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_flat_scaled = (
    input_scaler.fit_transform(
        X_train_flat
    )
)

X_test_flat_scaled = (
    input_scaler.transform(
        X_test_flat
    )
)

y_train_scaled = (
    target_scaler.fit_transform(
        y_train_sequence.reshape(-1, 1)
    ).ravel()
)

y_test_scaled = (
    target_scaler.transform(
        y_test_sequence.reshape(-1, 1)
    ).ravel()
)


# ================================================================
# PART C1: LINEAR AR(7)
# ================================================================

print("\nFitting linear AR(7)...")

linear_ar = LinearRegression()

linear_ar.fit(
    X_train_flat_scaled,
    y_train_scaled
)

linear_ar_predictions_scaled = (
    linear_ar.predict(
        X_test_flat_scaled
    )
)

linear_ar_predictions = (
    target_scaler.inverse_transform(
        linear_ar_predictions_scaled.reshape(-1, 1)
    ).ravel()
)

linear_ar_rmse = calculate_rmse(
    y_test_sequence,
    linear_ar_predictions
)

add_result(
    "C1",
    "Linear AR(7)",
    "cnt",
    "RMSE",
    linear_ar_rmse
)

print(
    f"Linear AR(7) test RMSE: "
    f"{linear_ar_rmse:.3f}"
)


# ================================================================
# PART C2: FEEDFORWARD NEURAL NETWORK
# ================================================================

print("\nFitting feedforward neural network...")

tf.random.set_seed(RANDOM_SEED)

feedforward_nn = models.Sequential(
    [
        layers.Input(
            shape=(
                X_train_flat_scaled.shape[1],
            )
        ),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.2),

        layers.Dense(
            32,
            activation="relu"
        ),

        layers.Dense(1)
    ]
)

feedforward_nn.compile(
    optimizer="adam",
    loss="mse"
)

early_stopping_ffnn = (
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )
)

history_ffnn = feedforward_nn.fit(
    X_train_flat_scaled,
    y_train_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping_ffnn],
    verbose=0,
    shuffle=False
)

feedforward_predictions_scaled = (
    feedforward_nn.predict(
        X_test_flat_scaled,
        verbose=0
    ).ravel()
)

feedforward_predictions = (
    target_scaler.inverse_transform(
        feedforward_predictions_scaled.reshape(-1, 1)
    ).ravel()
)

feedforward_rmse = calculate_rmse(
    y_test_sequence,
    feedforward_predictions
)

add_result(
    "C2",
    "Feedforward NN",
    "cnt",
    "RMSE",
    feedforward_rmse
)

print(
    f"Feedforward NN test RMSE: "
    f"{feedforward_rmse:.3f}"
)


# ------------------------------------------------
# FEEDFORWARD LEARNING CURVE
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history_ffnn.history["loss"],
    label="Training loss"
)

plt.plot(
    history_ffnn.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title(
    "Feedforward neural network learning curve"
)

plt.legend()
plt.grid(alpha=0.3)

save_current_plot(
    "C2_feedforward_NN_learning_curve.png"
)


# ================================================================
# RNN DATA STRUCTURE
# ================================================================

def construct_sequence_array(dataframe):

    sequences = []

    for _, row in dataframe.iterrows():

        demand_lags = [
            row[f"cnt_lag{lag}"]
            for lag in range(1, 8)
        ]

        temperature_lags = [
            row[f"temp_lag{lag}"]
            for lag in range(1, 8)
        ]

        windspeed_lags = [
            row[f"windspeed_lag{lag}"]
            for lag in range(1, 8)
        ]

        sequence = np.column_stack(
            [
                demand_lags,
                temperature_lags,
                windspeed_lags
            ]
        )

        sequences.append(sequence)

    return np.asarray(
        sequences,
        dtype=float
    )


X_train_sequence = (
    construct_sequence_array(
        train_lagged
    )
)

X_test_sequence = (
    construct_sequence_array(
        test_lagged
    )
)


# Scale each RNN channel using training data only.
X_train_sequence_scaled = np.zeros_like(
    X_train_sequence
)

X_test_sequence_scaled = np.zeros_like(
    X_test_sequence
)

sequence_scalers = []

for channel in range(3):

    channel_scaler = StandardScaler()

    training_channel = (
        X_train_sequence[:, :, channel]
    )

    testing_channel = (
        X_test_sequence[:, :, channel]
    )

    training_shape = training_channel.shape
    testing_shape = testing_channel.shape

    channel_scaler.fit(
        training_channel.reshape(-1, 1)
    )

    transformed_training = (
        channel_scaler.transform(
            training_channel.reshape(-1, 1)
        )
    )

    transformed_testing = (
        channel_scaler.transform(
            testing_channel.reshape(-1, 1)
        )
    )

    X_train_sequence_scaled[:, :, channel] = (
        transformed_training.reshape(
            training_shape
        )
    )

    X_test_sequence_scaled[:, :, channel] = (
        transformed_testing.reshape(
            testing_shape
        )
    )

    sequence_scalers.append(
        channel_scaler
    )


# ================================================================
# PART C3: SIMPLE RNN
# ================================================================

print("\nFitting simple RNN...")

tf.random.set_seed(RANDOM_SEED)

simple_rnn = models.Sequential(
    [
        layers.Input(
            shape=(7, 3)
        ),

        layers.SimpleRNN(
            32,
            activation="tanh"
        ),

        layers.Dense(
            16,
            activation="relu"
        ),

        layers.Dense(1)
    ]
)

simple_rnn.compile(
    optimizer="adam",
    loss="mse"
)

early_stopping_rnn = (
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )
)

history_rnn = simple_rnn.fit(
    X_train_sequence_scaled,
    y_train_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping_rnn],
    verbose=0,
    shuffle=False
)

rnn_predictions_scaled = (
    simple_rnn.predict(
        X_test_sequence_scaled,
        verbose=0
    ).ravel()
)

rnn_predictions = (
    target_scaler.inverse_transform(
        rnn_predictions_scaled.reshape(-1, 1)
    ).ravel()
)

rnn_rmse = calculate_rmse(
    y_test_sequence,
    rnn_predictions
)

add_result(
    "C3",
    "Simple RNN",
    "cnt",
    "RMSE",
    rnn_rmse
)

print(
    f"Simple RNN test RMSE: "
    f"{rnn_rmse:.3f}"
)


# ------------------------------------------------
# RNN LEARNING CURVE
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history_rnn.history["loss"],
    label="Training loss"
)

plt.plot(
    history_rnn.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title("Simple RNN learning curve")

plt.legend()
plt.grid(alpha=0.3)

save_current_plot(
    "C3_simple_RNN_learning_curve.png"
)


# ================================================================
# FINAL RESULTS
# ================================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

RESULTS["value"] = RESULTS[
    "value"
].astype(float)

RESULTS = RESULTS.sort_values(
    by="value",
    ascending=True
).reset_index(
    drop=True
)

print(
    RESULTS.to_string(index=False)
)


RESULTS.to_csv(
    os.path.join(
        FIG_DIR,
        "model_results.csv"
    ),
    index=False
)


# ================================================================
# BEST MODEL
# ================================================================

best_result = RESULTS.iloc[0]

print("\n================================================")
print("BEST MODEL")
print("================================================")

print(
    f"Best model: {best_result['model']}"
)

print(
    f"Best test RMSE: "
    f"{best_result['value']:.3f}"
)


# ================================================================
# MODEL COMPARISON PLOT
# ================================================================

plt.figure(figsize=(12, 7))

plot_results = RESULTS.sort_values(
    by="value",
    ascending=True
)

plt.barh(
    plot_results["model"],
    plot_results["value"],
    color="steelblue"
)

plt.xlabel("Test RMSE")
plt.ylabel("Model")
plt.title("Test RMSE comparison")
plt.grid(
    axis="x",
    alpha=0.3
)

save_current_plot(
    "final_model_comparison.png"
)


# ================================================================
# END
# ================================================================

print("\n================================================")
print("SCRIPT COMPLETE")
print("================================================")

print(
    "All figures and model_results.csv "
    "were saved in the figures folder."
)