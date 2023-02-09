'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
from models.BERTFineTuningDeployment import *

class BERTFieldExtractor(BERTFineTuningDeployment):
    """
    BERTFineTuningDeployment for field extraction
    """
    def __init__(self, redacted_input=False, max_len=512):
        """
        Initialization of BERTFineTuningDeployment
        :param redacted_input: bool, whether to predict on redacted data
        :param max_len: int, max sequence length used for the model
        """
        super(BERTFieldExtractor, self).__init__(redacted_input)
        self.max_len = max_len

    def setup_data_for_training(self, dt):
        """
        setp up of data and model for training.
        :param dt: list of dicts, data for training
        :return: list, of preprocessed data, [label, input tokens]
        """
        name_or_path = self.config._name_or_path
        self.model = BertForSequenceClassification_aux.from_pretrained(
            name_or_path, config=self.config)
        self.model.to(self.device)
        if all(['sectionized' in d.keys() for d in dt]):
            dset = [
                [self.config.label2id[dt[i][
                    self.config.field_extraction_config['field_name']
                ]],
                 self.tokenizer(dt[i]['sectionized'][
                                    self.config.field_extraction_config[
                                        'report_section']
                                ],
                                truncation=True)['input_ids']
                 ]
                for i in range(len(dt))
                if self.config.field_extraction_config['report_section'] in
                dt[i]['sectionized'].keys()]
        elif all([self.config.field_extraction_config['report_section'] in
                  d.keys() for d in dt]):
            dset = [
                [self.config.label2id[dt[i][
                    self.config.field_extraction_config['field_name']
                ]],
                 self.tokenizer(dt[i][
                                    self.config.field_extraction_config[
                                        'report_section']
                                ],
                                truncation=True)['input_ids']
                 ]
                for i in range(len(dt))
                if dt[i][self.config.field_extraction_config[
                             'report_section']]!='']
        else:
            raise Exception('the data set must have a sectioned column or a '
                            'column name matching the report_section')
        return dset

    def predict(self, x):
        """
        prediction on text x
        :param x: str
        :return: label of classification
        """
        prediction = self.model.forward(
            input_ids=torch.tensor(
                self.tokenizer(x, truncation=True)['input_ids']
            )[None, :].to(self.device)
        )['logits']
        return self.config.id2label[
            torch.softmax(
                prediction, dim=1
            ).argmax().item()
        ]
