import numpy as np
import pandas as pd
import pytest

from project.feature_enginnering.feature_engineering import PolynomialTransformer


def test_polynomial_transformer():
    # Sample DataFrame
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [0, 1, 2, 3],
            "non_numeric": ["a", "b", "c", "d"],
        }
    )

    # Initialize PolynomialTransformer
    transformer = PolynomialTransformer(columns=["feature1", "feature2"], degree=3)

    # Transform data
    transformed = transformer.fit_transform(data)

    # Expected Results
    expected_feature1_power2 = data["feature1"] ** 2
    expected_feature1_power3 = data["feature1"] ** 3
    expected_feature2_power2 = data["feature2"] ** 2
    expected_feature2_power3 = data["feature2"] ** 3

    # Assertions to check if new polynomial features are correct
    assert np.allclose(
        transformed["feature1_power2"], expected_feature1_power2
    ), "feature1_power2 failed!"
    assert np.allclose(
        transformed["feature1_power3"], expected_feature1_power3
    ), "feature1_power3 failed!"
    assert np.allclose(
        transformed["feature2_power2"], expected_feature2_power2
    ), "feature2_power2 failed!"
    assert np.allclose(
        transformed["feature2_power3"], expected_feature2_power3
    ), "feature2_power3 failed!"

    # Ensure non-numeric column is untouched
    assert np.all(
        transformed["non_numeric"] == data["non_numeric"]
    ), "Non-numeric column changed!"

    # Ensure original columns are retained
    assert "feature1" in transformed.columns, "Original feature1 missing!"
    assert "feature2" in transformed.columns, "Original feature2 missing!"

    # Print success
    print("PolynomialTransformer unit test passed successfully!")


# Run the test
if __name__ == "__main__":
    test_polynomial_transformer()
