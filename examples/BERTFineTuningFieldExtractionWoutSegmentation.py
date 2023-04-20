'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py

Description: Training code used to train a model on the identification of a
field in the input text. This is meant to train a classifier WITHOUT section
segmentation as a pre processing step.

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
from utils import TrainingUtils as cutl
from models.BERTFieldExtractorWoutST import BERTFieldExtractorWoutST
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

cutl.set_seed(20210429)

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
parser.add_argument('--k_fold', action='store_true')
parser.add_argument("--use_redacted", action='store_true')
parser.add_argument('--field_name', type=str, required=True,
                    help='Field of interest to be extracted from reports.')
opt = parser.parse_args()

print('-'*80)
print(opt)
print('-'*80)

# Set up data for training and testing
data = cutl.load_all_data(opt.dfolder)

Labels = list(set([row[opt.field_name] for row in data]))
Labels.sort()
if opt.k_fold:
    folds = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(
            data,
            [Labels.index(row[opt.field_name]) for row in data]
    ):
        train, test = [data[i] for i in train_index], \
                      [data[i] for i in test_index]
        folds.append(
            {'train': train, 'test': test}
        )
    print('Beginning K-Fold training Experiment')
else:
    train, test = cutl.train_test_spliting(
        data,
        [Labels.index(row[opt.field_name]) for row in data]
    )
    folds = [{'train': train, 'test': test}]

# Begin Experiment
results = {'subject': [],
           'input': [],
           'GT': [],
           'PR': []}

for k, fold in enumerate(folds):
    print('Running the ' + str(k+1) + ' kfold experiment.')
    # Initialize Model
    labeler = BERTFieldExtractorWoutST(redacted_input=opt.use_redacted)

    labeler.from_pretrained(opt.pre_trained_model)

    labeler.config.num_labels = len(Labels)
    labeler.config.label2id = {lbl: i for i, lbl in enumerate(Labels)}
    labeler.config.id2label = {i: lbl for i, lbl in enumerate(Labels)}
    labeler.config.field_extraction_config = {'field_name': opt.field_name}

    # Begin Training
    labeler.fine_tune(fold['train'],
                      max_epochs=opt.n_epochs,
                      batch_size=opt.batch_size)
    labeler.save_model(opt.sfolder + str(k))

    labeler = BERTFieldExtractorWoutST(redacted_input=opt.use_redacted)
    labeler.from_pretrained(opt.sfolder + str(k))

    # Begin Testing

    test_res = []
    incorrect = []
    correct_ex = []
    acc = []
    ct = 0
    print('Amount of Test Subjects: ' + str(len(fold['test'])))

    for report in fold['test']:
        results['subject'].append(report['filename'])
        results['GT'].append(report[opt.field_name])
        results['input'].append(report['original_report'])
        results['PR'].append(labeler.predict(x=report['original_report']))

# Summarize Testing Results
metrics = {'Accuracy': [],
           'G.F1': [],
           'Weighted_F1': [],
           'Weighted_precision': [],
           'Weighted_recall': []}
metrics = cutl.calculate_testing_metrics(results['PR'], results['GT'], metrics)

test_results = pd.DataFrame(results)
summary_metrics = pd.DataFrame(metrics)

print('Accuracy of BERTFieldExtractorWoutST(): ' + str(np.round(
    metrics['Accuracy'][0]*100)))
print('G.F1 of BERTFieldExtractorWoutST(): ' + str(np.round(
    metrics['G.F1'][0]*100)))
print('-' * 60)

# list of dataframes and sheet names
dfs = [summary_metrics, test_results]
sheets = ['Summary_Metrics', 'Test_results']

# run function
cutl.dfs_tabs(dfs, sheets, opt.sfolder + '_test_results.xlsx')
print('End of BERTFineTuningFieldExtractionWoutSectionization.py Script')
