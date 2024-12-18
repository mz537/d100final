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
from sklearn.metrics import auc, mean_squared_error
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
    "Recency",
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
train_loss_glm_1 = gaussian_deviance(
    y_train_t, df_train["pp_glm1"], w_train_t
) / np.sum(w_train_t)
test_loss_glm_1 = gaussian_deviance(y_test_t, df_test["pp_glm1"], w_test_t) / np.sum(
    w_test_t
)
print(f"Training loss (Glm1, Gaussian Deviance): {train_loss_glm_1: .5f}")
print(f"Testing loss (Glm1, Gaussian Deviance): {test_loss_glm_1: .5f}")

# %%
# add splines for numerical variables and use pipeline

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

preprocessor.set_output(transform="pandas")  # Output as DataFrame
model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(family="gaussian", fit_intercept=True),
        ),
    ]
)

model_pipeline

# check if the pipeline works
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)
# %%
pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_glm2"] = model_pipeline.predict(df_test)
df_train["pp_glm2"] = model_pipeline.predict(df_train)

train_loss_glm_2 = gaussian_deviance(
    y_train_t, df_train["pp_glm2"], w_train_t
) / np.sum(w_train_t)
test_loss_glm_2 = gaussian_deviance(y_test_t, df_test["pp_glm2"], w_test_t) / np.sum(
    w_test_t
)
print(f"Training loss (Glm2, Gaussian Deviance): {train_loss_glm_2: .5f}")
print(f"Testing loss (Glm2, Gaussian Deviance): {test_loss_glm_2: .5f}")


# %%
# we can see that deviance has declined by using splines and pipelines
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
train_loss_gbm = gaussian_deviance(y_train_t, df_train["pp_gbm"], w_train_t) / np.sum(
    w_train_t
)
test_loss_gbm = gaussian_deviance(y_test_t, df_test["pp_gbm"], w_test_t) / np.sum(
    w_test_t
)

# Print results
print(f"Training loss (GBM, Gaussian Deviance): {train_loss_gbm: .5f}")
print(f"Testing loss (GBM, Gaussian Deviance): {test_loss_gbm: .5f}")

# Compare total observed vs predicted on test set
print(
    "Total income on test set, observed = {:.2f}, predicted = {:.2f}".format(
        y_test_t.sum(), np.sum(df_test["pp_gbm"])
    )
)
# %%
# reduce overfitting by tuning the pipeline

cv = GridSearchCV(
    model_pipeline_lgbm,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "estimate__n_estimators": [100, 200, 500],
    },
    verbose=2,
)
cv.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)
lgbm_unconstrained = cv.best_estimator_

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

train_loss_lgbm = gaussian_deviance(
    y_train_t, df_train["pp_t_lgbm"], w_train_t
) / np.sum(w_train_t)
test_loss_lgbm = gaussian_deviance(y_test_t, df_test["pp_t_lgbm"], w_test_t) / np.sum(
    w_test_t
)

# Print results
print(f"Training loss (LGBM, Gaussian Deviance): {train_loss_lgbm: .5f}")
print(f"Testing loss (LGBM, Gaussian Deviance): {test_loss_lgbm: .5f}")

# Compare total observed vs predicted on test set
print(
    "Total income on test set, observed = {:.2f}, predicted = {:.2f}".format(
        y_test_t.sum(), np.sum(df_test["pp_t_lgbm"])
    )
)

# %%

df_plot_fish = (
    df.groupby("MntFishProducts_log")
    .apply(lambda x: np.average(x["Income"]))
    .reset_index(name="Income")
)
sns.scatterplot(df_plot_fish, x="MntFishProducts_log", y="Income")

# %%
df_plot_meat = (
    df.groupby("MntMeatProducts_log")
    .apply(lambda x: np.average(x["Income"]))
    .reset_index(name="Income")
)
sns.scatterplot(df_plot_meat, x="MntMeatProducts_log", y="Income")
# %%
df_plot_ren = (
    df.groupby("Recency")
    .apply(lambda x: np.average(x["Income"]))
    .reset_index(name="Income")
)
sns.scatterplot(df_plot_ren, x="Recency", y="Income")
# %%

df_plot_gold = (
    df.groupby("MntGoldProds_log")
    .apply(lambda x: np.average(x["Income"]))
    .reset_index(name="Income")
)
sns.scatterplot(df_plot_gold, x="MntGoldProds_log", y="Income")

# %%
df_plot_wine = (
    df.groupby("MntWines_log")
    .apply(lambda x: np.average(x["Income"]))
    .reset_index(name="Income")
)
sns.scatterplot(df_plot_wine, x="MntWines_log", y="Income")

# %%
df_plot_fruit = (
    df.groupby("MntFruits_log")
    .apply(lambda x: np.average(x["Income"]))
    .reset_index(name="Income")
)
sns.scatterplot(df_plot_fruit, x="MntFruits_log", y="Income")
# %%
# Hyperparameter Tuning for LGBM with Monotonic Constraints
lgbm_constrained = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="regression",
                monotone_constraints=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            ),
        )
    ]
)

cv = GridSearchCV(
    lgbm_constrained,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.05],
        "estimate__n_estimators": [500, 1000],
    },
    verbose=2,
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

train_loss_clgbm = gaussian_deviance(
    y_train_t, df_train["pp_t_clgbm"], w_train_t
) / np.sum(w_train_t)
test_loss_clgbm = gaussian_deviance(y_test_t, df_test["pp_t_clgbm"], w_test_t) / np.sum(
    w_test_t
)

# Print results
print(f"Training loss (LGBM, Gaussian Deviance): {train_loss_clgbm: .5f}")
print(f"Testing loss (LGBM, Gaussian Deviance): {test_loss_clgbm: .5f}")

lgb.plot_metric(lgbm_constrained[0])

train_mse_clgbm = mean_squared_error(
    y_train_t, df_train["pp_t_clgbm"], sample_weight=w_train_t
)
test_mse_clgbm = mean_squared_error(
    y_test_t, df_test["pp_t_clgbm"], sample_weight=w_test_t
)
print(f"Best parameters: {cv.best_params_}")
print(f"Training MSE: {train_mse_clgbm: .5f}")
print(f"Testing MSE: {test_mse_clgbm: .5f}")

# %%
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
shap = lgbm_constrained_exp.predict_parts(X_test_t.head(1), type="shap")

shap.plot()
# %%
# Lorenz Curve
# Plot Lorenz Curve for Models
fig, ax = plt.subplots(figsize=(8, 8))

# Add Lorenz Curve for Gaussian Model Predictions
for label, y_pred in [
    ("Gaussian LGBM", df_test["pp_t_lgbm"]),  # unconstrained LGBM model
    ("Constraint LGBM ", df_test["pp_t_clgbm"]),  # constrained LGBM model
    ("GLM Splines", df_test["pp_glm2"]),  # GLM with Splines
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
