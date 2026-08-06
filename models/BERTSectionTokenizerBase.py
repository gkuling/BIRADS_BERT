'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute

Shared implementation for the paper's two section-segmentation arms
(BERTSectionTokenizer and BERTSectionTokenizerWithAux).
'''
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import PretrainedConfig, BertTokenizerFast

from models.BERTFineTuningDeployment import BERTFineTuningDeployment
from models.BertForSequenceClassification_aux import \
    BertForSequenceClassification_aux
from utils.TextPreprocessing import report_preprocess, \
    determine_report_GT, get_sents_and_redacted, gt_preprocessing


class BERTSectionTokenizerBase(BERTFineTuningDeployment):
    """
    Base class for the two section tokenizers. Subclasses define
    _encode_example (the shape of one training record) and
    _predict_logits (how sentence logits are produced at inference).
    """

    # The order of this list defines label2id/id2label, which save_model
    # persists into config.json - reordering it silently invalidates every
    # previously saved checkpoint.
    SECTION_LABELS = ['PrIM',
                      'Procedure',
                      'Ethics',
                      'Dx',
                      'Title',
                      'Findings',
                      'HX',
                      'Impression'
                      ]
    AUX_DATA_SIZE = 0

    def __init__(self, redacted_input=False):
        """
        Initializer for the section tokenizer
        :param redacted_input: bool, choice to use the model with redacted data
        """
        super(BERTSectionTokenizerBase, self).__init__(
            redacted_input, aux_data=self.AUX_DATA_SIZE)
        self.max_len = 32

    def _encode_example(self, ex, ln_num, total):
        """
        Build one training record from a [label, sentence, previous label]
        example.
        :param ex: [label, sentence text, previous sentence label]
        :param ln_num: int, sentence number within the report
        :param total: int, total number of sentences in the report
        :return: list, one training record
        """
        raise NotImplementedError()

    def _predict_logits(self, encodings, pred_on):
        """
        Produce one logits tensor per encoded sentence.
        :param encodings: list of input_ids lists, one per sentence
        :param pred_on: list of sentences being predicted on
        :return: list of logits tensors
        """
        raise NotImplementedError()

    def setup_data_for_training(self, dt):
        """
        setup data and model for training
        :param dt: list of dicts, data for training
        :return: dict of training and validation lists of preprocessed data
        """
        print('Running Data Preparation for Training')
        labels = list(self.SECTION_LABELS)
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
                    sectionized = {key: gt_preprocessing(report['sectionized'][key])
                                   for key in report['sectionized'].keys()
                                   if key in self.config.label2id.keys()}
                else:
                    sectionized = {
                        key: gt_preprocessing(report[key])
                        for key in labels
                        if key in self.config.label2id.keys() and report[key]!=''}

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
                add_on = [self._encode_example(ex, ln_num,
                                               len(gt_and_sent_and_aux))
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

        prediction = self._predict_logits(encodings, pred_on)
        label_out = [self._logits_to_label(lbl) for lbl in prediction]

        output = {}
        for i, lbl in enumerate(label_out):
            if lbl in output.keys():
                output[lbl] = output[lbl] + ' ' + orig_sents[i]
            else:
                output[lbl] = orig_sents[i]
        if 'Dx' in output.keys():
            output['Dx'] = output['Dx'].replace('.', ';')
        return output, label_out
