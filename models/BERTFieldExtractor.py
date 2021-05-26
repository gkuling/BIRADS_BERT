from models.BERTFineTuningDeployment import *

class BERTFieldExtractor(BERTFineTuningDeployment):
    def __init__(self, redacted_input=False, max_len=512):
        super(BERTFieldExtractor, self).__init__(redacted_input)
        self.max_len = max_len

    def setup_data_for_training(self, dt):
        name_or_path = self.config._name_or_path
        self.model = BertForSequenceClassification_aux.from_pretrained(
            name_or_path, config=self.config)
        self.model.to(self.device)

        dset = [[self.config.label2id[dt[i][
            self.config.field_extraction_config['field_name']
                                      ]],
                 self.tokenizer(dt[i]['sectionized'][
            self.config.field_extraction_config['report_section']
                                       ],
                                truncation=True)['input_ids']]
                for i in range(len(dt)) if
                self.config.field_extraction_config['report_section'] in
                dt[i]['sectionized'].keys()]
        return dset

    def predict(self, x):
        prediction = self.model.forward(
            input_ids=torch.tensor(
                self.tokenizer(x, truncation=True)['input_ids']
            )[None,:].to(self.device)
        )['logits']
        return self.config.id2label[
            torch.softmax(
                prediction, dim=1
            ).argmax().item()
        ]
