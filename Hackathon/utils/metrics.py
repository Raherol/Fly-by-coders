import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(prediction, target):
    '''
    Returns Root Mean Squared Error
    '''
    return np.sqrt(mean_squared_error(prediction, target))


def ioa(prediction, target):
    '''
    Returns Index of Agreement
    '''
    prediction = np.array(prediction).reshape(-1)
    target = np.array(target).reshape(-1)

    mean_observed = np.mean(target)

    numerator = np.sum((prediction - target) ** 2)

    denominator = np.sum(
        (np.abs(prediction - mean_observed) + np.abs(target - mean_observed)) ** 2)

    ioa = 1 - (numerator / denominator)

    return ioa


def ioa_debugger(prediction, target):
    '''
    Returns Index of Agreement
    '''
    prediction = np.array(prediction).reshape(-1)

    print(f'FROM DEBUGGER: prediction = {prediction}')

    target = np.array(target).reshape(-1)

    print(f'FROM DEBUGGER: target = {target}')

    mean_observed = np.mean(target)

    print(f'FROM DEBUGGER: mean = {mean_observed}')

    numerator = np.sum((prediction - target) ** 2)

    print(f'FROM DEBUGGER: numerator = {numerator}')

    denominator = np.sum(
        (np.abs(prediction - mean_observed) + np.abs(target - mean_observed)) ** 2)

    print(f'FROM DEBUGGER: denominator = {denominator}')

    ioa = 1 - (numerator / denominator)

    print(f'FROM DEBUGGER: ioa = {ioa}')


def bias(prediction, target):
    '''
    Returns Bias
    '''
    prediction = np.array(prediction).reshape(-1)
    target = np.array(target).reshape(-1)

    n = len(prediction)

    bias = np.sum(prediction - target)/n

    return bias


def extract_last_elements(df, model):
    last_elements = df.groupby(['year', 'section']).tail(1)[model].values
    return last_elements


def calculate_rmse(df, pred_model):
    last_predictions = extract_last_elements(df, pred_model)
    last_targets = extract_last_elements(df, 'meters')

    rmse_value = rmse(last_predictions, last_targets)

    return rmse_value


def calculate_bias(df, pred_model):
    last_predictions = extract_last_elements(df, pred_model)
    last_targets = extract_last_elements(df, 'meters')

    rmse_value = bias(last_predictions, last_targets)

    return rmse_value


def compute_score(*, y_prediction_seconds: list, y_true_seconds: list):
    """Compute score using predicted values and real values.

    The goal is to MINIMIZE the score.
    Predictions above the real value are penalized more than predictions below the real value.

    Args:
        y_prediction_seconds: list of predictions in seconds.
        y_true_seconds: list of real values in seconds.
    """
    if len(y_prediction_seconds) != len(y_true_seconds):
        raise ValueError("The length of the lists must be the same")
    individual_scores = []
    for y_prediction, y_true in zip(y_prediction_seconds, y_true_seconds, strict=True):
        delta = y_prediction - y_true
        if delta >= 0:
            score = pow(abs(delta), 2)
        else:
            score = pow(abs(delta), 1)
        individual_scores.append(score)
    final_score = sum(individual_scores)
    return final_score
