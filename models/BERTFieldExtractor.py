'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
'''
from models.BERTFieldExtractorBase import BERTFieldExtractorBase


class BERTFieldExtractor(BERTFieldExtractorBase):
    """
    BERTFineTuningDeployment for field extraction. Twin of
    BERTFieldExtractorWoutST: this arm classifies a single report section
    (config.field_extraction_config['report_section']) produced by section
    segmentation, where its twin classifies the whole preprocessed report.
    """

    def _text_label_pairs(self, dt):
        """
        Pair each record's configured report section with its field label.
        Records may carry sections inside a 'sectionized' dict or as
        top-level columns.
        :param dt: list of dicts, data for training
        :return: list of (text, label) tuples
        """
        section = self.config.field_extraction_config['report_section']
        if all(['sectionized' in d.keys() for d in dt]):
            return [(d['sectionized'][section], self._field_label(d))
                    for d in dt
                    if section in d['sectionized'].keys()]
        elif all([section in d.keys() for d in dt]):
            return [(d[section], self._field_label(d))
                    for d in dt
                    if d[section]!='']
        else:
            raise Exception('the data set must have a sectioned column or a '
                            'column name matching the report_section')

    def _encode_input(self, x):
        """
        The input is already a report section; tokenize it as is.
        :param x: str
        :return: str
        """
        return x
