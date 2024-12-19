# %%
import dalex as dx
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import auc, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from project.data._sample_split import create_sample_split
from project.data.preprocessing import load_mcdata_p
from project.evaluation._evaluate_predictions import (
    evaluate_predictions,
    gaussian_deviance,
    lorenz_curve,
)

# %%
# load cleaned data
file_path = "../project/data/marketing_campaign_cleaned.parquet"
df = load_mcdata_p(file_path)
df.head()


# %%
# train a glm model with gaussian family
weight = np.ones(len(df))  # here we set equal weight
y = df["Income"]

df = create_sample_split(df, id_column="ID")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")

df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

print(df_train.dtypes)


# %%
categoricals = [
    "Education",
    "Marital_Status",
    "Age",
    "Children",
    "NumStorePurchases",
    "NumWebPurchases",
    "NumWebVisitsMonth",
    "NumDealsPurchases",
]


numeric_cols = [
    "MntWines_log",
    "MntFruits_log",
    "MntMeatProducts_log",
    "MntGoldProds_log",
    "MntFishProducts_log",
]
predictors = categoricals + numeric_cols
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]
# %%
pp_glm1 = GeneralizedLinearRegressor(
    family="gaussian", drop_first=True, fit_intercept=True
)
pp_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

df_test["pp_glm1"] = pp_glm1.predict(X_test_t)
df_train["pp_glm1"] = pp_glm1.predict(X_train_t)


pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="Income",
    preds_column="pp_glm1",
)

# %%
# try gamma family
pp_glm3 = GeneralizedLinearRegressor(
    family="gamma", drop_first=True, fit_intercept=True
)
pp_glm3.fit(X_train_t, y_train_t, sample_weight=w_train_t)

df_test["pp_glm3"] = pp_glm3.predict(X_test_t)
df_train["pp_glm3"] = pp_glm3.predict(X_train_t)


pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="Income",
    preds_column="pp_glm3",
)

# the result from gaussian is better than gamma, so we choose gaussian


# %%
# hypertuning GLM model for alpha and l1
# Custom scorer for Gaussian Deviance
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Preprocessing: Splines for numeric columns, OneHot for categoricals
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("spline", SplineTransformer(include_bias=False, knots="quantile")),
                ]
            ),
            numeric_cols,
        ),
        ("categorical", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

# Define the pipeline with preprocessing and GLM estimator
glm_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("estimate", GeneralizedLinearRegressor(family="gaussian", fit_intercept=True)),
    ]
)

# Define parameter grid for hyperparameter tuning
param_grid = {
    "estimate__alpha": [0.01, 0.1, 1],  # Regularization strength
    "estimate__l1_ratio": [0, 0.25, 0.5, 0.75, 1],  # Mix of L1 (Lasso) and L2 (Ridge)
}

# Grid search with cross-validation
grid_search_glm = GridSearchCV(
    estimator=glm_pipeline,
    param_grid=param_grid,
    scoring=scorer,
    cv=10,  # 10-fold cross-validation
    verbose=2,
)

# Fit the grid search on training data
grid_search_glm.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

# Extract the best pipeline and parameters
best_glm_pipeline = grid_search_glm.best_estimator_
print("Best parameters:", grid_search_glm.best_params_)

# Evaluate the best pipeline on the test set
df_test["pp_best_glm_pipeline"] = best_glm_pipeline.predict(df_test)
df_train["pp_best_glm_pipeline"] = best_glm_pipeline.predict(df_train)

test_loss_best_glm_pipeline = mean_absolute_error(
    y_test_t, df_test["pp_best_glm_pipeline"]
)
print(f"Test MAE: {test_loss_best_glm_pipeline: .5f}")

# we can see that Best parameters: {'estimate__alpha': 1, 'estimate__l1_ratio': 1}


# %%

# now lets use GBM as estimator
model_pipeline_lgbm = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="regression",
                learning_rate=0.1,
                n_estimators=1000,
                num_leaves=6,
                early_stopping_rounds=25,
            ),
        )  # Adjust learning rate and number of trees,early stopping
    ]
)

# Fit the GBM model
model_pipeline_lgbm.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)

# Add predictions to train and test sets
df_train["pp_gbm"] = model_pipeline_lgbm.predict(X_train_t)
df_test["pp_gbm"] = model_pipeline_lgbm.predict(X_test_t)


# Calculate training and testing loss
pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="Income",
    preds_column="pp_gbm",
)

# %%
# hyperparameter tuning for pipeline of unconstrained lgbm

cv_u = GridSearchCV(
    model_pipeline_lgbm,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "estimate__n_estimators": [1000],
        "estimate__num_leaves": [6, 12, 24],  # Leaf nodes per tree
        "estimate__min_child_weight": [1, 5, 10],
    },
    cv=5,
    scoring=scorer,
    verbose=2,
)
cv_u.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)
lgbm_unconstrained = cv_u.best_estimator_

df_test["pp_t_lgbm"] = cv_u.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv_u.best_estimator_.predict(X_train_t)

pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="Income",
    preds_column="pp_t_lgbm",
)
# %%
print("Best Parameters for unconstrained LGBM:")
print(cv_u.best_params_)
# Best Parameters for unconstrained LGBM:
# {'estimate__learning_rate': 0.05, 'estimate__min_child_weight': 1,
# 'estimate__n_estimators': 1000, 'estimate__num_leaves': 6

# %%
# Hyperparameter Tuning for LGBM with Monotonic Constraints
# from EDA we know that numerical features have positive relationships
# we set monotone_constraints for them as 1 and 0 for categoricals
lgbm_constrained = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="regression",
                monotone_constraints=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            ),
        )
    ]
)

cv = GridSearchCV(
    lgbm_constrained,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "estimate__n_estimators": [1000],
        "estimate__num_leaves": [6, 12, 24],  # Leaf nodes per tree
        "estimate__min_child_weight": [1, 5, 10],
    },
    verbose=2,
    cv=5,
    scoring=scorer,
)

cv.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)
df_test["pp_t_clgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_clgbm"] = cv.best_estimator_.predict(X_train_t)
lgbm_constrained.fit(
    X_train_t,
    y_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)


lgb.plot_metric(lgbm_constrained[0])
pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="Income",
    preds_column="pp_t_clgbm",
)
# %%
print("Best Parameters for Constrained LGBM:")
print(cv.best_params_)
# Best Parameters for Constrained LGBM:
# {'estimate__learning_rate': 0.01, 'estimate__min_child_weight': 1,
#  'estimate__n_estimators': 1000, 'estimate__num_leaves': 12}


# %%
# partial dependency plots
lgbm_constrained_exp = dx.Explainer(
    lgbm_constrained, X_test_t, y_test_t, label="Constrained LGBM"
)
pdp_constrained = lgbm_constrained_exp.model_profile()

lgbm_unconstrained_exp = dx.Explainer(
    lgbm_unconstrained, X_test_t, y_test_t, label="Unconstrained LGBM"
)
pdp_unconstrained = lgbm_unconstrained_exp.model_profile()

pdp_constrained.plot(pdp_unconstrained)

# %%
# shapley values plots
shap = lgbm_constrained_exp.predict_parts(X_test_t.head(1), type="shap")


shap.plot()


# %%
# pdp for top 5 features from constrained LGBM
# top 5 features from constrained LGBM
top_5_features = [
    "MntWines_log",
    "MntMeatProducts_log",
    "Children",
    "NumStorePurchases",
    "NumWebPurchases",
]

# Plot Partial Dependence for the top 5 features
fig, ax = plt.subplots(nrows=5, figsize=(12, 20))

PartialDependenceDisplay.from_estimator(
    lgbm_constrained.named_steps[
        "estimate"
    ],  # Extracting the LGBMRegressor from the pipeline
    X_test_t,  # Test dataset
    features=top_5_features,
    grid_resolution=50,
    ax=ax,
)

plt.tight_layout()
plt.show()

# %%
# Predict vs Actual Plot for constrained, unconstrained LGBM and tuned GLM
models_predictions = {
    "Constrained LGBM": df_test["pp_t_clgbm"],
    "Unconstrained LGBM": df_test["pp_t_lgbm"],
    "Tuned GLM Pipeline": df_test["pp_best_glm_pipeline"],
}
actual_values = y_test_t

# Plot Predict vs. Actual for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (model_name, predictions) in zip(axes, models_predictions.items()):
    ax.scatter(actual_values, predictions, alpha=0.5)
    ax.plot(
        [actual_values.min(), actual_values.max()],
        [actual_values.min(), actual_values.max()],
        "r--",
    )
    ax.set_title(f"{model_name}\nPredict vs. Actual")
    ax.set_xlabel("Actual Income")
    ax.set_ylabel("Predicted Income")
    ax.grid(True)

plt.tight_layout()
plt.show()
# %%
# Lorenz Curve
# Plot Lorenz Curve for Models
fig, ax = plt.subplots(figsize=(8, 8))

# Add Lorenz Curve for Gaussian Model Predictions
for label, y_pred in [
    ("Gaussian LGBM", df_test["pp_t_lgbm"]),  # unconstrained LGBM model
    ("Constraint LGBM ", df_test["pp_t_clgbm"]),  # constrained LGBM model
    ("GLM Splines", df_test["pp_best_glm_pipeline"]),  # GLM with Splines
    ("GLM Benchmark", df_test["pp_glm1"]),  # GLM Benchmark
]:
    ordered_samples, cum_income = lorenz_curve(
        y_true=df_test["Income"], y_pred=y_pred, weight=w_train_t
    )
    gini = 1 - 2 * auc(ordered_samples, cum_income)  # Calculate Gini index
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_income, linestyle="-", label=label)

# Oracle Model: Perfect predictions (y_pred = y_true)
ordered_samples, cum_income = lorenz_curve(
    y_true=df_test["Income"], y_pred=df_test["Income"], weight=w_test_t
)
gini = 1 - 2 * auc(ordered_samples, cum_income)
ax.plot(
    ordered_samples,
    cum_income,
    linestyle="-.",
    color="gray",
    label=f"Oracle (Gini: {gini: .3f})",
)

# Random Baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")

# Finalize the plot
ax.set(
    title="Lorenz Curve for Income Predictions",
    xlabel="Fraction of observations\n(ordered by model from lowest to highest income)",
    ylabel="Fraction of total income",
)
ax.legend(loc="upper left")
plt.show()

# %%
