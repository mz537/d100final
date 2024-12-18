import matplotlib.pyplot as plt
import seaborn as sns


def plot_numerical_distribution(df, column_name, bins=30):
    """
    Plot the distribution of a numerical column with a histogram and KDE.

    Parameters:
    - df (DataFrame): The dataset.
    - column_name (str): The column to plot.
    - bins (int): Number of bins for the histogram.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column_name], bins=bins, kde=True, color="skyblue")
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def plot_categorical_count(df, column_name):
    """
    Plot the count of each category in a categorical column.

    Parameters:
    - df (DataFrame): The dataset.
    - column_name (str): The column to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column_name, data=df, palette="muted")
    plt.title(f"Count of Categories in {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()


def plot_correlation_heatmap(df, figsize=(10, 8)):
    """
    Plot a heatmap to show correlations between numerical features.

    Parameters:
    - df (DataFrame): The dataset (numerical columns only).
    - figsize (tuple): Size of the heatmap.

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    numerical_df = df.select_dtypes(include=["int64", "float64"])
    correlation_matrix = numerical_df.corr()

    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Heatmap")
    plt.show()


def plot_reboxplot(df, column_name, target_column):
    """
    Plot a boxplot to show the relationship between a categorical column and a numerical target.

    Parameters:
    - df (DataFrame): The dataset.
    - column_name (str): The categorical column.
    - target_column (str): The numerical column to analyze.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=column_name, y=target_column, data=df, palette="Set3")
    plt.title(f"Boxplot of {target_column} by {column_name}")
    plt.xlabel(column_name)
    plt.ylabel(target_column)
    plt.xticks(rotation=45)
    plt.show()


def plot_scatter(df, x_column, y_column):
    """
    Plot a scatter plot to show the relationship between two numerical columns.

    Parameters:
    - df (DataFrame): The dataset.
    - x_column (str): Column for the x-axis.
    - y_column (str): Column for the y-axis.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x_column, y=y_column, data=df, color="darkblue")
    plt.title(f"Scatter Plot of {y_column} vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid()
    plt.show()


def plot_single_boxplot(df, column_name):
    """
    Plot a single boxplot to visualize the distribution of a numerical column.

    Parameters:
    - df (DataFrame): The dataset.
    - column_name (str): The numerical column to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=column_name, data=df, color="lightgreen")
    plt.title(f"Boxplot of {column_name}")
    plt.ylabel(column_name)
    plt.grid()
    plt.show()
