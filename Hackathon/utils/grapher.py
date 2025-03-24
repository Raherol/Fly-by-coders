import matplotlib.pyplot as plt
from utils import metrics


def plot_metrics_bar(metric_values, model_names, metric_name):
    """
    Generate a bar plot for a given metric across different models.

    Parameters:
        metric_values (list): List of metric values for each model.
        model_names (list): List of model names corresponding to metric values.
        metric_name (str): Name of the metric (e.g., "RMSE", "IoA").
    """
    fig, ax = plt.subplots()

    ax.bar(model_names, metric_values)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} across Different Models')

    if metric_name == 'IoA':
        plt.ylim(0.9, 1.0)

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_metrics_bar_v2(metric_values, model_names, metric_name):
    """
    Generate a bar plot for a given metric across different models.

    Parameters:
        metric_values (list): List of metric values for each model.
        model_names (list): List of model names corresponding to metric values.
        metric_name (str): Name of the metric (e.g., "RMSE", "IoA").
    """
    fig, ax = plt.subplots()

    ax.bar(model_names, metric_values)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} feature importance')

    if metric_name == 'IoA':
        plt.ylim(0.9, 1.0)

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
