"""
Response parsing utilities for GLUE benchmark tasks.
These functions convert model generated text responses into the correct label formats.
"""

from typing import Union

def parse_generated_text(task_name: str, text: str) -> Union[int, float]:
    """Parse generated text to extract prediction.
    
    Args:
        task_name: Name of the GLUE task
        text: Generated text from the model
        
    Returns:
        Parsed prediction as an integer or float
    """
    text = text.strip().lower()
    
    if task_name == "cola":
        if "yes" in text or "correct" in text or "acceptable" in text or "grammatical" in text:
            return 1
        else:
            return 0
    
    elif task_name in ["mnli", "mnli-mm"]:
        if "entailment" in text:
            return 2
        elif "contradiction" in text:
            return 0
        else:  # neutral or anything else
            return 1
    
    elif task_name in ["mrpc", "qqp"]:
        if "yes" in text or "equivalent" in text or "paraphrase" in text or "same" in text:
            return 1
        else:
            return 0
    
    elif task_name == "qnli":
        if "yes" in text or "entail" in text:
            return 0  # entailment
        else:
            return 1  # not entailment
    
    elif task_name == "rte":
        if "entail" in text or "yes" in text:
            return 0  # entailment
        else:
            return 1  # not entailment
    
    elif task_name == "sst2":
        if "positive" in text:
            return 1
        else:
            return 0
    
    elif task_name == "stsb":
        # Try to extract a number from the text
        try:
            for word in text.split():
                if word.replace('.', '').isdigit():
                    score = float(word)
                    # Normalize to 0-5 range if needed
                    if score > 5:
                        score = 5.0
                    if score < 0:
                        score = 0.0
                    return score
            # Default to middle value if no number found
            return 2.5
        except ValueError:
            return 2.5
    
    elif task_name == "wnli":
        if "yes" in text or "entail" in text:
            return 1
        else:
            return 0
    
    else:
        raise ValueError(f"Task {task_name} not supported")