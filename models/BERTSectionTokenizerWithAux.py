'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
import sys
sys.path.append('.')

from models.BERTFineTuningDeployment import *
from utils.TextPreprocessing import report_preprocess, \
    determine_report_GT, get_sents_and_redacted, gt_preprocessing
import numpy as np

class BERTSectionTokenizerWithAux(BERTFineTuningDeployment):
    """
    Bert Section tokenizer with Auxiliary data
    """
    def __init__(self, redacted_input=False):
        """
        Initialization for BERT Fine Tuning Deployment
        :param redacted_input: bool
        """
        super(BERTSectionTokenizerWithAux, self).__init__(redacted_input,
                                                          aux_data=3)
        self.redacted_input = redacted_input
        self.max_len = 32

    def setup_data_for_training(self, dt):
        """
        setup data and model for training
        :param dt: list of dicts, data for training
        :return: list, of preprocessed data, [label, input tokens, auxiliary
            data: [prev sent label, sent number, total number of sent]
        """
        print('Running Data Preparation for Training')
        labels = ['PrIM',
                  'Procedure',
                  'Ethics',
                  'Dx',
                  'Title',
                  'Findings',
                  'HX',
                  'Impression'
                  ]
        aux_data = self.config.aux_data_size
        name_or_path = self.config._name_or_path
        self.config = PretrainedConfig.from_pretrained(name_or_path)
        self.config.aux_data_size = aux_data
        self.config.num_labels = len(labels)
        self.config.label2id = {lbl:i for i,lbl in enumerate(labels)}
        self.config.id2label = {i:lbl for i,lbl in enumerate(labels)}


        self.model = BertForSequenceClassification_aux.from_pretrained(
            name_or_path, config=self.config)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            name_or_path)
        self.tokenizer.model_max_length = self.max_len
        self.model.to(self.device)
        tr, dv = train_test_split(dt, stratify=[d['Modality'] for d in dt],
                                  test_size=0.15, random_state=20210330)
        dset_split = {'training': tr,
                      'val': dv}
        for key in dset_split.keys():
            dt_Temp = dset_split[key]
            dset = []
            for report in tqdm(dt_Temp):
                if 'sectionized' in report.keys():
                    sectionized = {
                        key: gt_preprocessing(report['sectionized'][key])
                        for key in report['sectionized'].keys()
                        if key in self.config.label2id.keys()}
                else:
                    sectionized = {
                        key: gt_preprocessing(report[key])
                        for key in labels
                        if key in self.config.label2id.keys() and report[
                            key] != ''}
                processed_data = report_preprocess(report['original_report'])
                sents, orig_sents = get_sents_and_redacted(processed_data)
                GT = determine_report_GT(orig_sents,
                                         sectionized,
                                         self.config.label2id.keys())
                if self.redacted_input:
                    pred_on = sents
                else:
                    pred_on = orig_sents
                gt_and_sent = [[GT[i], pred_on[i]] for i in range(len(GT))]
                gt_and_sent_and_aux = [
                    [gt_and_sent[i][0], gt_and_sent[i][1], gt_and_sent[i-1][0]]
                    for i in range(1, len(gt_and_sent))
                ]
                gt_and_sent_and_aux.insert(0,
                    [gt_and_sent[0][0], gt_and_sent[0][1], 'Start'])
                add_on = [[self.config.label2id[ex[0]],
                           self.tokenizer(ex[1], truncation=True)['input_ids'],
                           [
                            self.config.label2id[ex[2]]
                            if ex[2] in self.config.label2id.keys()
                            else -1,
                            ln_num,
                            len(gt_and_sent_and_aux)]
                           ]
                          for ln_num, ex in enumerate(gt_and_sent_and_aux)]


                dset.extend(add_on)
            dset_split[key] = dset
        return dset_split

    def predict(self, x):
        """
        prediction on x
        :param x: str, raw text report
        :return: dict, of the report split into sections
        """
        self.model.eval()
        processed_data = report_preprocess(x)
        sents, orig_sents = get_sents_and_redacted(processed_data)
        if self.redacted_input:
            pred_on = sents
        else:
            pred_on = orig_sents

        encodings = [self.tokenizer(snt, truncation=True)['input_ids'] for snt
                     in pred_on]
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
        label_out = [self.config.id2label[
                         torch.softmax(lbl, dim=-1).argmax(-1).item()
                     ]
                     for lbl in prediction]

        output = {}
        for i, lbl in enumerate(label_out):
            if lbl in output.keys():
                output[lbl] = output[lbl] + ' ' + orig_sents[i]
            else:
                output[lbl] = orig_sents[i]
        if 'Dx' in output.keys():
            output['Dx'] = output['Dx'].replace('.', ';')
        return output, label_out