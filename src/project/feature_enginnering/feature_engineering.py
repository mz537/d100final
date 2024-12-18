import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PolynomialTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate polynomial features for specified columns.
    """

    def __init__(self, columns=None, degree=2):
        self.columns = columns
        self.degree = degree

    def fit(self, X, y=None):
        self.columns_ = (
            self.columns if self.columns else X.select_dtypes(include=np.number).columns
        )
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns_:
            for d in range(2, self.degree + 1):
                X_transformed[f"{col}_power{d}"] = X_transformed[col] ** d
        return X_transformed
