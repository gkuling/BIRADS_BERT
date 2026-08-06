'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute

Shared implementation for the paper's two field-extraction arms
(BERTFieldExtractor and BERTFieldExtractorWoutST).
'''
import torch

from models.BERTFineTuningDeployment import BERTFineTuningDeployment
from models.BertForSequenceClassification_aux import \
    BertForSequenceClassification_aux


class BERTFieldExtractorBase(BERTFineTuningDeployment):
    """
    Base class for the two field extractors. Subclasses define
    _text_label_pairs (how raw records become (text, label) training
    pairs) and _encode_input (how a raw prediction input is prepared
    before tokenization).
    """
    def __init__(self, redacted_input=False, max_len=512):
        """
        Initialization of BERTFieldExtractorBase
        :param redacted_input: bool, whether to predict on redacted data
        :param max_len: int, max sequence length used for the model
        """
        super(BERTFieldExtractorBase, self).__init__(redacted_input)
        self.max_len = max_len

    def _text_label_pairs(self, dt):
        """
        Turn raw records into (text, label) training pairs. Must return a
        list, not a generator, so input validation raises before any
        tokenization work starts.
        :param dt: list of dicts, data for training
        :return: list of (text, label) tuples
        """
        raise NotImplementedError()

    def _encode_input(self, x):
        """
        Prepare one raw prediction input for tokenization.
        :param x: str
        :return: str
        """
        raise NotImplementedError()

    def _field_label(self, record):
        """
        Pull the classification label for the configured field from a record.
        :param record: dict for one report
        :return: label value
        """
        return record[self.config.field_extraction_config['field_name']]

    def setup_data_for_training(self, dt):
        """
        set up of data and model for training.
        :param dt: list of dicts, data for training
        :return: list, of preprocessed data, [label, input tokens]
        """
        name_or_path = self.config._name_or_path
        self.model = BertForSequenceClassification_aux.from_pretrained(
            name_or_path, config=self.config)
        self.model.to(self.device)
        dset = [[self.config.label2id[label],
                 self.tokenizer(text, truncation=True)['input_ids']]
                for text, label in self._text_label_pairs(dt)]
        return dset

    def predict(self, x):
        """
        prediction on text x
        :param x: str
        :return: label of classification
        """
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(
                input_ids=torch.tensor(
                    self.tokenizer(self._encode_input(x),
                                   truncation=True)['input_ids']
                )[None, :].to(self.device)
            )['logits']
        return self._logits_to_label(prediction)
