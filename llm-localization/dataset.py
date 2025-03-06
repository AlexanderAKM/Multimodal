import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset

from utils import read_story, read_question

class MDLocDataset(Dataset):
    """
    A synthetic dataset that generates arithmetic questions (addition and subtraction)
    with two levels of difficulty:
      - Positive examples: Numbers between 100 and 200.
      - Negative examples: Numbers between 1 and 20.
    """
    def __init__(self):  
        """
        Initializes the dataset by generating `num_examples` arithmetic problems 
        for both positive and negative examples.
        """
        num_examples = 100

        self.positive = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(100, 200)
            num_2 = np.random.randint(100, 200)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.positive.append(f"Question: {question}\nAnswer: {answer}")
            
        self.negative = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(1, 20)
            num_2 = np.random.randint(1, 20)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.negative.append(f"Question: {question}\nAnswer: {answer}")

    def __getitem__(self, idx):
        """
        Returns the positive and negative example at the given index.
        
        Args:
            idx (int): Index of the dataset item.
        
        Returns:
            tuple: (positive example, negative example) as formatted strings.
        """
        return self.positive[idx].strip(), self.negative[idx].strip()
        
    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.positive) 


class VisLocDataset(Dataset):
    pass

class LangLocDataset(Dataset):
    """
    A dataset that processes language localization stimuli from CSV files.
    The dataset builds a vocabulary and provides positive and negative samples 
    based on predefined stimulus categories.
    """
    def __init__(self):
        """
        Loads data from multiple CSV files, processes text stimuli,
        builds vocabulary, and categorizes data into positive and negative samples.
        """
        dirpath = "stimuli/language"
        paths = glob(f"{dirpath}/*.csv")
        vocab = set()

        # Load and merge data from all CSV files
        data = pd.read_csv(paths[0])
        for path in paths[1:]:
            run_data = pd.read_csv(path)
            data = pd.concat([data, run_data])

        # Process text stimuli (convert to lowercase)
        data["sent"] = data["stim2"].apply(str.lower)
        vocab.update(data["stim2"].apply(str.lower).tolist())
        
        for stimuli_idx in range(3, 14):
            data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)
            vocab.update(data[f"stim{stimuli_idx}"].apply(str.lower).tolist())

        # Create word-index mappings
        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        # Categorize sentences based on `stim14` label
        self.positive = data[data["stim14"] == "S"]["sent"]
        self.negative = data[data["stim14"] == "N"]["sent"]

        # Limit the number of examples to 5, just for now as running time otherwise takes too long!
        self.positive = self.positive.iloc#[:5]
        self.negative = self.negative.iloc#[:5]

    def __getitem__(self, idx):
        """
        Returns the positive and negative example at the given index.
        
        Args:
            idx (int): Index of the dataset item.
        
        Returns:
            tuple: (positive sentence, negative sentence) as strings.
        """
        return self.positive.iloc[idx].strip(), self.negative.iloc[idx].strip()
        
    def __len__(self):
        """
        Returns the total number of positive examples.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.positive)

class TOMLocDataset(Dataset):
    """
    A dataset for Theory of Mind (ToM) localization. It loads and formats
    stories and questions related to belief-based and photograph-based tasks.
    """
    def __init__(self):
        """
        Loads and processes stories and questions from text files,
        categorizing them as positive (belief-based) or negative (photograph-based).
        """
        dirpath = "stimuli/tom/tomloc"
        instruction = "In this experiment, you will read a series of sentences and then answer True/False questions about them. Press button 1 to answer 'true' and button 2 to answer 'false'."
        context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: {answer}"
        
        
        # Load stories and questions
        belief_stories = [read_story(f"{dirpath}/{idx}b_story.txt") for idx in range(1, 11)]
        photograph_stories = [read_story(f"{dirpath}/{idx}p_story.txt") for idx in range(1, 11)]

        belief_questions = [read_question(f"{dirpath}/{idx}b_question.txt") for idx in range(1, 11)]
        photograph_questions = [read_question(f"{dirpath}/{idx}p_question.txt") for idx in range(1, 11)]

        # Create positive (belief-based) and negative (photo-based) examples
        self.positive = [context_template.format(instruction=instruction, story=story, question=question, answer=np.random.choice(["True", "False"])) for story, question in zip(belief_stories, belief_question)]
        self.negative = [context_template.format(instruction=instruction, story=story, question=question, answer=np.random.choice(["True", "False"])) for story, question in zip(photograph_stories, photograph_question)]


    def __getitem__(self, idx):
        """
        Returns the positive and negative example at the given index.
        """
        return self.positive[idx].strip(), self.negative[idx].strip()
    
    def __len__(self):
        """
        Returns the total number of positive examples.
        """
        return len(self.positive)
    
class ImageLocDataset(Dataset):
    def _init__(self):
        pass
