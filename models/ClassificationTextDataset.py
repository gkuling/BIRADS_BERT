'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
from torch.utils.data import Dataset

class ClasificationTextDataset(Dataset):
    """
    Dataset used for text classification with auxiliary data
    """
    def __init__(self, data, aux_data=False):
        """
        Initilaization of Dataset
        :param data: list of data
        :param aux_data: bool for auxiliary data is in data
        """
        self.examples = data
        self.aux_data = aux_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        if self.aux_data:
            return {'input_ids': example[1],
                    'labels': example[0],
                    'aux_data': example[2]}
        else:
            return {'input_ids': example[1],
                    'labels': example[0]}