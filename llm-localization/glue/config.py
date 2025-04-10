"""
GLUE benchmark configuration settings
"""

# GLUE task configurations
GLUE_TASKS = {
    "cola": {
        "metric": "matthews_correlation",
        "num_labels": 2,
    },
    "mnli": {
        "metric": "accuracy",
        "num_labels": 3,
    },
    "mnli-mm": {
        "metric": "accuracy",
        "num_labels": 3,
    },
    "mrpc": {
        "metric": "f1",
        "num_labels": 2,
    },
    "qnli": {
        "metric": "accuracy",
        "num_labels": 2,
    },
    "qqp": {
        "metric": "f1",
        "num_labels": 2,
    },
    "rte": {
        "metric": "accuracy",
        "num_labels": 2,
    },
    "sst2": {
        "metric": "accuracy",
        "num_labels": 2,
    },
    "stsb": {
        "metric": "spearmanr",
        "num_labels": 1,
    },
    "wnli": {
        "metric": "accuracy",
        "num_labels": 2,
    }
}