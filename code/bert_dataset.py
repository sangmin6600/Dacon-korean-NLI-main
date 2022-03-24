import torch
from torch.utils.data import Dataset


class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        premises = [s['premise'] for s in samples]
        hypothesises = [s['hypothesis'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            premises,
            hypothesises,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['premise'] = premises
            return_value['hypothesis'] = hypothesises

        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, premises, hypothesises, labels):
        self.premises = premises
        self.hypothesises = hypothesises
        self.labels = labels
    
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, item):
        premise = str(self.premises[item])
        hypothesis = str(self.hypothesises[item])
        label = self.labels[item]

        return {
            'premise' : premise,
            'hypothesis' : hypothesis,
            'label' : label
        }
