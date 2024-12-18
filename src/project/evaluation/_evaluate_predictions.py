import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error


def evaluate_predictions(
    df,
    outcome_column,
    *,
    preds_column=None,
    model=None,
    weights_column=None,
    family="gaussian",
):
    """Evaluate predictions against actual outcomes.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe used for evaluation
    outcome_column : str
        Name of the actual outcome column
    preds_column : str, optional
        Name of the predictions column, by default None
    model : object, optional
        Fitted model, by default None
    exposure_column : str, optional
        Name of the exposure column for weighted evaluation, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation metrics (MSE, RMSE, MAE, Bias, Gini index)
    """
    evals = {}

    assert preds_column or model, ""

    # Generate predictions if preds_column is not provided
    if preds_column is None:
        preds = model.predict(df)
    else:
        preds = df[preds_column]

    # Handle exposure weights
    if weights_column:
        weights = df[weights_column]
    else:
        weights = np.ones(len(df))

    # Weighted averages for bias calculation
    evals["mean_preds"] = np.average(preds, weights=weights)
    evals["mean_outcome"] = np.average(df[outcome_column], weights=weights)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    # Error Metrics
    evals["mse"] = np.average((preds - df[outcome_column]) ** 2, weights=weights)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(preds - df[outcome_column]), weights=weights)

    # Lorenz Curve and Gini Index
    ordered_samples, cum_actuals = lorenz_curve(df[outcome_column], preds, weights)
    evals["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)

    # Deviance Calculation
    if family == "gaussian":
        # Deviance for Gaussian is Residual Sum of Squares
        evals["deviance"] = np.sum(weights * (df[outcome_column] - preds) ** 2)
    elif family == "poisson":
        # Deviance for Poisson
        evals["deviance"] = 2 * np.sum(
            weights
            * (
                df[outcome_column]
                * np.log(np.maximum(df[outcome_column] / preds, 1e-10))
                - (df[outcome_column] - preds)
            )
        )
    elif family == "binomial":
        # Deviance for Binomial
        evals["deviance"] = 2 * np.sum(
            weights
            * (
                df[outcome_column]
                * np.log(np.maximum(df[outcome_column] / preds, 1e-10))
                + (1 - df[outcome_column])
                * np.log(np.maximum((1 - df[outcome_column]) / (1 - preds), 1e-10))
            )
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    return pd.DataFrame(evals, index=[0]).T


def lorenz_curve(y_true, y_pred, weight):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    weight = np.asarray(weight)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_weight = weight[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_weight)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


def gaussian_deviance(y_true, y_pred, sample_weight):
    residuals = y_true - y_pred
    return np.sum(sample_weight * residuals**2)
