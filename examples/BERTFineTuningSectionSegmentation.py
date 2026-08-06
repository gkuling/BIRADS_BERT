'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py

Description: Training code used to train a model on the identification of a
section being described in the input text. This code can be used to train the
section tokenizer with and without auxiliary data.

Input: labeled data as a folder of .txt documents holding dictionaries
    with the input data and field label, or a single .csv/.xlsx file.
Output: A test_results.xlsx file with the individual test predictions and
ground truth, and summary stats of accuracy and G. F1 Score. Saved
pytorch model (pytorch_model.bin, config.json), and tokenizer (
special_tokens_map.json, tokenizer_config.json, and vocab.txt).
'''
import sys
sys.path.append('.')

import numpy as np
import pandas as pd

from models.BERTSectionTokenizer import BERTSectionTokenizer
from models.BERTSectionTokenizerWithAux import BERTSectionTokenizerWithAux
from utils import ExperimentUtils as exutl
from utils import TrainingUtils as cutl
from utils.TextPreprocessing import report_preprocess, \
    determine_report_GT, get_sents_and_redacted, gt_preprocessing

cutl.set_seed(20210501)

parser = exutl.build_common_parser()
parser.add_argument("--aux_data", action='store_true',
                    help="Train the section tokenizer with auxiliary data.")
parser.add_argument("--data_portion", type=float, default=1.0,
                    help="Portion of data to be used for training. ")
opt = parser.parse_args()

print('-'*80)
print(opt)
print('-'*80)

# Set up data for training and testing
data = cutl.load_all_data(opt.data_location)
folds, _ = exutl.build_folds(data, 'Modality', k_fold=opt.k_fold)

# Begin Experiment
metrics = cutl.new_metrics_dict()
for k, fold in enumerate(folds):
    print('Running the ' + str(k+1) + ' kfold experiment.')
    # Initialize Model
    if opt.aux_data:
        sec_tknzr = BERTSectionTokenizerWithAux(redacted_input=opt.use_redacted)
    else:
        sec_tknzr = BERTSectionTokenizer(redacted_input=opt.use_redacted)
    sec_tknzr.from_pretrained(opt.pre_trained_model)

    # Begin Training
    sec_tknzr.fine_tune(fold['train'],
                        max_epochs=opt.n_epochs,
                        batch_size=opt.batch_size,
                        ablation_portion=opt.data_portion)
    sec_tknzr.save_model(opt.sfolder + str(k))

    if opt.aux_data:
        sec_tknzr = BERTSectionTokenizerWithAux(redacted_input=opt.use_redacted)
    else:
        sec_tknzr = BERTSectionTokenizer(redacted_input=opt.use_redacted)
    sec_tknzr.from_pretrained(opt.sfolder + str(k))

    # Begin Testing
    # Metrics are computed once per report, so Test_results holds one row
    # per report: post_ExperimentSummary.py's Mann-Whitney branch consumes
    # exactly this per-report distribution.
    print('Amount of Test Subjects: ' + str(len(fold['test'])))
    for report in fold['test']:
        if 'sectionized' in report.keys():
            sectionized = {key: gt_preprocessing(report['sectionized'][key])
                           for key in report['sectionized'].keys()
                           if key in sec_tknzr.config.label2id.keys()}
        else:
            sectionized = {
                key: gt_preprocessing(report[key])
                for key in sec_tknzr.config.label2id.keys()
                if key in sec_tknzr.config.label2id.keys() and report[key] != ''}
        processed_data = report_preprocess(report['original_report'])
        sents, orig_sents = get_sents_and_redacted(processed_data)
        GT = determine_report_GT(orig_sents,
                                 sectionized,
                                 sec_tknzr.config.label2id.keys())
        pr_sectionized, lbl = sec_tknzr.predict(
            report['original_report']
        )
        metrics = cutl.calculate_testing_metrics(lbl, GT, metrics)

test_results = pd.DataFrame(metrics)

# Summarize Testing Results
summary_metrics = cutl.new_metrics_dict()
metrics = test_results.columns
for metric in metrics:
    ave = test_results[
        metric].mean()
    sdv = test_results[
        metric].std()
    summary_metrics[metric].extend([ave, sdv])
print('Accuracy of BERTSectionTokenizer(): ' +
      str(np.round(summary_metrics['Accuracy'][0]*100)) + '+/-' +
      str(np.round(summary_metrics['Accuracy'][1]*100)) + '%')
print('G.F1 of BERTSectionTokenizer(): ' +
      str(np.round(summary_metrics['G.F1'][0]*100)) + '+/-' +
      str(np.round(summary_metrics['G.F1'][1]*100)) + '%')
print('-' * 60)
summary_metrics = pd.DataFrame(summary_metrics)

exutl.write_experiment_results(summary_metrics, test_results, opt.sfolder)
print('End of BERTFineTuningSectionSegmentation.py Script')
