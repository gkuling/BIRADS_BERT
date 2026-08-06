'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
import numpy as np
import torch

from models.BERTSectionTokenizerBase import BERTSectionTokenizerBase


class BERTSectionTokenizerWithAux(BERTSectionTokenizerBase):
    """
    Bert Section tokenizer with Auxiliary data. Twin of
    BERTSectionTokenizer: identical data preparation, but each training
    record carries an auxiliary vector of [previous sentence label,
    sentence number, total sentences], and prediction is autoregressive -
    sentence i's auxiliary vector uses the prediction for sentence i-1.
    """
    AUX_DATA_SIZE = 3

    def _encode_example(self, ex, ln_num, total):
        """
        Build a [label id, input tokens, auxiliary vector] training record.
        :param ex: [label, sentence text, previous sentence label]
        :param ln_num: int, sentence number within the report
        :param total: int, total sentences in the report
        :return: list, one training record
        """
        return [self.config.label2id[ex[0]],
                self.tokenizer(ex[1], truncation=True)['input_ids'],
                [
                 self.config.label2id[ex[2]]
                 if ex[2] in self.config.label2id.keys()
                 else -1,
                 ln_num,
                 total]
                ]

    def _predict_logits(self, encodings, pred_on):
        """
        Predict sentences autoregressively: the first sentence gets a
        'Start' marker (-1), every later sentence's auxiliary vector uses
        the previous sentence's predicted label.
        :param encodings: list of input_ids lists, one per sentence
        :param pred_on: list of sentences being predicted on
        :return: list of logits tensors
        """
        prediction = [self.model(
            input_ids=torch.tensor(np.array(
                encodings[0])[None,:]
            ).to(self.device),
            aux_data=torch.tensor(
                [
                     [
                         -1,
                         0,
                         len(pred_on)
                     ]
                ]
            ).type(torch.FloatTensor).to(self.device)
        )['logits'] ]

        for i in range(1, len(encodings)):
            prediction.extend([self.model(
                input_ids=torch.tensor(np.array(
                    encodings[i])[None,:]
                ).to(self.device),
                aux_data=torch.tensor(
                    [
                        [
                            torch.softmax(prediction[i - 1], dim=-1).argmax(
                            -1).item(),
                            i,
                            len(pred_on)
                        ]
                    ]
                ).type(torch.FloatTensor).to(self.device)
            )['logits']])
        return prediction
