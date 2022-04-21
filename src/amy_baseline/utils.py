"""Helper functions for calculating evaluation metrics."""

import re

def exact_match_ratio(y_true, y_pred):
    """Calculate exact match ratio."""
    n = len(y_true)
    exact_match_count = 0
    for tru, pre in zip(y_true, y_pred):
        if tru == pre:
            exact_match_count += 1
    return exact_match_count / n


def empty_response_rate(pred_list):
    """Calculate empty response rate."""
    empty = len([x for x in pred_list if x == []])
    return empty / len(pred_list)

