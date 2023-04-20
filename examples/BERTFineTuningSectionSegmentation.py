'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py

Description: Training code used to train a model on the identification of a
section being described in the input text. This code can be used to train the
section tokenizer with and without auxiliary data.

Input: train and test folders filled with .txt documents holding dictionaries
    with the input data and field label.
Output: A test_results.xlsx file with the individual test predictions and
ground truth, and summary stats of accuracy and G. F1 Score. Saved
pytorch model (pytorch_model.bin, config.json), and tokenizer (
special_tokens_map.json, tokenizer_config.json, and vocab.txt).
'''
import sys
sys.path.append('.')

import argparse
import utils.TrainingUtils as cutl
from models.BERTSectionTokenizer import BERTSectionTokenizer
from models.BERTSectionTokenizerWithAux import BERTSectionTokenizerWithAux
from utils.TextPreprocessing import report_preprocess, \
    determine_report_GT, get_sents_and_redacted, gt_preprocessing
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

cutl.set_seed(20210501)

parser = argparse.ArgumentParser()
parser.add_argument("--data_location", type=str,
                    help="folder to find experimental data. Must have a train"
                         " and test folder.")
parser.add_argument("--sfolder", type=str,
                    help="folder to save data")
parser.add_argument('--pre_trained_model', type=str,
                    default='bert-base-uncased',
                    help="The directory of pretrained model or the pretrained "
                         "weights from huggingface.")
parser.add_argument("--n_epochs", type=int, default=4,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--aux_data", action='store_true')
parser.add_argument("--use_redacted", action='store_true')
parser.add_argument("--data_portion", type=float, default=1.0,
                    help="Portion of data to be used for training. ")
parser.add_argument("--k_fold", action='store_true',
                    help="Choice to run the experiment in a 5 kfold validation "
                         "experiment. ")
opt = parser.parse_args()

print('-'*80)
print(opt)
print('-'*80)

# Set up data for training and testing
data = cutl.load_all_data(opt.dfolder)

modLabels = list(set([row['Modality'] for row in data]))
modLabels.sort()
if opt.k_fold:
    folds = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(data,
                                             [modLabels.index(row['Modality'])
                                              for row in data]):
        train, test = [data[i] for i in train_index], \
                      [data[i] for i in test_index]
        folds.append(
            {'train': train, 'test': test}
        )
    print('Beginning K-Fold training Experiment')
else:
    train, test = cutl.train_test_spliting(data,
                                           [modLabels.index(row['Modality'])
                                            for row in data])
    folds = [{'train': train, 'test': test}]

# Begin Experiment
metrics = {'Accuracy': [],
           'G.F1': [],
           'Weighted_F1': [],
           'Weighted_precision': [],
           'Weighted_recall': []}
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

    test_res = []
    incorrect = []
    correct_ex = []
    acc = []
    ct = 0
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
summary_metrics = {'Accuracy': [],
                   'G.F1': [],
                   'Weighted_F1': [],
                   'Weighted_precision': [],
                   'Weighted_recall': []}
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

# list of dataframes and sheet names
dfs = [summary_metrics, test_results]
sheets = ['Summary_Metrics', 'Test_results']

# run function
cutl.dfs_tabs(dfs, sheets, opt.sfolder + '_test_results.xlsx')
print('End of BERTFineTuningSectionSegmentation.py Script')
