import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression


def plot_correlation_with_target(X, y, target_col="log_cnt", save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the target column.

    Args:
    - X (pd.DataFrame): DataFrame containing the data, not including the target column.
    - y (pd.DataFrame): DataFrame containing the target column. Must have the same amount of observation of X.
    - target_col (str, optional): Name of the target column. If not specified, the name "num" will be used.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - fig: The figure to plot in the notebook
    """

    # Check that the dataframes are aligned
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")
    
    # Concat the dataframes
    df = pd.concat([X, y], axis=1)

    # Compute the correlation with the target column
    correlations = df.corr()[target_col].drop(target_col).sort_values()
    
    # Configure the color palette
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Create a discrete color palette
    colors = sns.diverging_palette(10, 130, n=len(correlations))

    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(correlations.index, correlations.values, color=colors)

    # Configure the title and axis names
    plt.title(f"Correlation with {target_col.title()}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig


def plot_information_gain_with_target(X, y, target_col="log_cnt", save_path=None):
    """
    Plots the information gain of each variable in the dataframe with the target column.

    Args:
    - X (pd.DataFrame): DataFrame containing the data, not including the target column.
    - y (pd.DataFrame): DataFrame containing the target column. Must have the same amount of observation of X.
    - target_col (str, optional): Name of the target column. If not specified, the name "num" will be used.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - fig: The figure to plot in the notebook
    """

    # Check that the dataframes are aligned
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")

    # Compute the information gain with the target column for regression
    importances = pd.Series(mutual_info_regression(X, y.to_numpy().ravel()), X.columns).sort_values()

    # Configure the color palette
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Create a color palette
    colors = sns.diverging_palette(10, 130, n=len(importances))

    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(importances.index, importances, color=colors)

    # Configure the title and axis names
    plt.title(f"Information Gain with {target_col.title()}", fontsize=18)
    plt.xlabel("Information Gain", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig