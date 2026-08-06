'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
from models.BERTFieldExtractorBase import BERTFieldExtractorBase
from utils import TextPreprocessing as TPreP


class BERTFieldExtractorWoutST(BERTFieldExtractorBase):
    """
    BERTFineTuningDeployment for Field Extraction without section
    segmentation. Twin of BERTFieldExtractor: this arm classifies the whole
    preprocessed report, where its twin classifies a single report section
    produced by section segmentation.
    """

    def _text_label_pairs(self, dt):
        """
        Pair each record's preprocessed full report with its field label.
        :param dt: list of dicts, data for training
        :return: list of (text, label) tuples
        """
        return [(TPreP.report_preprocess(d['original_report'])['rep_in_para'],
                 self._field_label(d))
                for d in dt]

    def _encode_input(self, x):
        """
        Preprocess the raw report before tokenization.
        :param x: str, raw report text
        :return: str, report collapsed into paragraphs
        """
        return TPreP.report_preprocess(x)['rep_in_para']
