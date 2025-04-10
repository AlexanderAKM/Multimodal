"""
Utility functions for GLUE benchmark evaluation
"""

import logging
import os
from typing import Dict, List, Optional, Union, Tuple

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration for GLUE evaluation

    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("glue_benchmark")

def save_results(results: Dict, save_dir: str, task_name: str):
    """Save evaluation results to disk

    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save results
        task_name: Name of the GLUE task
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_name}_results.txt")
    with open(save_path, "w") as f:
        for metric_name, value in results.items():
            f.write(f"{task_name} - {metric_name}: {value:.4f}\n")
    return save_path