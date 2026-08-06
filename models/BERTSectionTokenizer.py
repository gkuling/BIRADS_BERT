'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
import torch

from models.BERTSectionTokenizerBase import BERTSectionTokenizerBase


class BERTSectionTokenizer(BERTSectionTokenizerBase):
    """
    BERTFineTuning Deployment for section segmentation without auxiliary
    data. Twin of BERTSectionTokenizerWithAux: identical data preparation,
    but training records carry no auxiliary vector and prediction runs each
    sentence independently rather than autoregressively.
    """
    AUX_DATA_SIZE = 0

    def _encode_example(self, ex, ln_num, total):
        """
        Build a [label id, input tokens] training record.
        :param ex: [label, sentence text, previous sentence label]
        :param ln_num: int, sentence number within the report (unused here)
        :param total: int, total sentences in the report (unused here)
        :return: list, one training record
        """
        return [self.config.label2id[ex[0]],
                self.tokenizer(ex[1], truncation=True)['input_ids']
                ]

    def _predict_logits(self, encodings, pred_on):
        """
        Predict each sentence independently.
        :param encodings: list of input_ids lists, one per sentence
        :param pred_on: list of sentences being predicted on (unused here)
        :return: list of logits tensors
        """
        return [self.model(
            input_ids=torch.tensor(
                x_enc
            )[None,:].to(self.device)
        )['logits'] for x_enc in encodings]
