'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
from models.BERTFineTuningDeployment import *
from utils import TextPreprocessing as TPreP


class BERTFieldExtractorWoutST(BERTFineTuningDeployment):
    """
    BERTFineTuningDeployment for Field Extraction without section tokenization
    """
    def __init__(self, redacted_input=False, max_len=512):
        """
        Initializer for BERTFieldExtractor without Section tokenization
        :param redacted_input: bool, whether to use redacted data
        :param max_len: int, max length of input sequence
        """
        super(BERTFieldExtractorWoutST, self).__init__(redacted_input)
        self.max_len = max_len

    def setup_data_for_training(self, dt):
        """
        set up of training data and model for training
        :param dt: list of dicts, data for training
        :return: list, of preprocessed data, [label, input tokens]
        """
        name_or_path = self.config._name_or_path
        self.model = BertForSequenceClassification_aux.from_pretrained(
            name_or_path, config=self.config)
        self.model.to(self.device)

        dset = [
            [self.config.label2id[dt[i][
                self.config.field_extraction_config['field_name']
            ]],
             self.tokenizer(
                 TPreP.report_preprocess(dt[i]['original_report'])[
                     'rep_in_para'],
                 truncation=True
             )['input_ids']]
            for i in range(len(dt))]
        return dset

    def predict(self, x):
        """
        prediction on x
        :param x: str
        :return: label of classification
        """
        prediction = self.model.forward(
            input_ids=torch.tensor(
                self.tokenizer(
                    TPreP.report_preprocess(x)['rep_in_para'],
                    truncation=True
                )['input_ids']
            )[None, :].to(self.device)
        )['logits']
        return self.config.id2label[
            torch.softmax(
                prediction, dim=1
            ).argmax().item()
        ]
