'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute

Description: this script will preprocessing the pretraining data into a .txt
file that can be used for pre training the BERT embedding.

Input: a .csv file that holds the reports for pretraining.
output: .txt files of pretraining data with vocabulary size, and a validation set.
'''
import sys
sys.path.append('.')

import argparse
import os
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from utils.TextPreprocessing import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dfolder", type=str,
                    help="Folder with CSV files to be processed.")
parser.add_argument("--ft_folder", type=str,
                    help="Folder with TXT files for fine tuning tasks.")
parser.add_argument("--exam_col", type=str, default="AccNum",
                    help="Column name used for patient ID.")
parser.add_argument("--date_col", type=str, default="ExamDate",
                    help="Column name used for exam date.")
parser.add_argument("--txt_col", type=str, default="ReportTxt",
                    help="Column name used for the report text.")
opt = parser.parse_args()

print('-'*80)
print(opt)
print('-'*80)

# Load all data files
databases = [pd.read_csv(opt.dfolder + '/' + fl)
             for fl in os.listdir(opt.dfolder) if fl.endswith('.csv')]
databases = [db[[opt.exam_col, opt.date_col, opt.txt_col]].copy() for db in
             databases]
all_data = pd.concat(databases, ignore_index=True)
all_data = all_data.drop_duplicates(opt.txt_col)

print('Total amount of reports: ')
print(all_data.shape)

# Take out Fine Tuning Data
val_set = [opt.ft_folder + '/' + fl for fl in os.listdir(opt.ft_folder) if
           fl.endswith('txt')]
print('Pulling out validation set.')
print(len(val_set))
for file in val_set:
    with open(file, 'r') as text:
        data = text.read()

    data = eval(data)
    all_data = all_data.loc[all_data[opt.txt_col] != data[
        'original_report']].copy()
print('Final Pretraining Amount of reports: ')
print(all_data.shape)
print('Beginning Preprocessing and saving Pretraining data. 10% for '
      'validation, 90% for training.')
training_csv, val_csv = train_test_split(all_data, test_size=0.10,
                                         random_state=20210325)
prp_data = {}
sets = ['training', 'validation']
for i1, dt_set in enumerate([training_csv, val_csv]):
    print('')
    print('Preprocessing ' + sets[i1])
    cumulative_vocab = []
    redacted_cumulative_vocab = []
    cumulative_words = 0
    train_data = []
    redacted_train_data = []
    for rp in tqdm(dt_set[opt.txt_col].tolist()):
        try:
            txt_processed = report_preprocess(rp)
            txt = txt_processed['rep_in_para']
            if 'add_in_para' in txt_processed.keys():
                txt += '\n' + txt_processed['add_in_para']
            train_data.append(txt)
            rtxt = txt_processed['rep_in_para_redacted']
            if 'add_in_para_redacted' in txt_processed.keys():
                rtxt += '\n' + txt_processed['add_in_para_redacted']
            redacted_train_data.append(rtxt)
        except:
            print('A Report was abandoned.' + str(rp[opt.exam_col]) + ' ' +
                  str(rp[opt.date_col]))
            continue
        words = [item for sublist in
                 [word_tokenize(snt) for snt in sent_tokenize(txt)]
                 for item in sublist]
        cumulative_words += len(words)
        vocab = list(set(words))
        words = [item for sublist in
                 [word_tokenize(snt) for snt in sent_tokenize(rtxt)]
                 for item in sublist]
        rvocab = list(set(words))
        cumulative_vocab.extend(vocab)
        redacted_cumulative_vocab.extend(rvocab)
        cumulative_vocab = list(set(cumulative_vocab))
        redacted_cumulative_vocab = list(set(redacted_cumulative_vocab))

    print('Vocabulary size for unredacted data:')
    print(len(cumulative_vocab))
    print('Vocabulary size for redacted data:')
    print(len(redacted_cumulative_vocab))
    print('Total Words in set:')
    print(cumulative_words)
    prp_data.update({sets[i1]: train_data})
    prp_data.update({sets[i1] + '_vocab': len(cumulative_vocab)})

print('')
print('Writing TXT files')
for dt_set in [key for key in prp_data.keys() if '_vocab' not in key]:
    if 'training' in dt_set:
        output_dir = opt.dfolder + \
                     '/VocabOf' + \
                     str(prp_data[dt_set + '_vocab']) + \
                     '_PreTraining_' + \
                     dt_set + '.txt'
    else:
        output_dir = opt.dfolder + '/PreTraining_' + dt_set + '.txt'
    f = open(output_dir, 'w')
    for rp in tqdm(prp_data[dt_set]):

        if len(rp) > 1:
            f.write(rp + '\n')
        else:
            continue
    f.close()
    print('Saved ' + output_dir)
    print('')

print("End of TextPreProcessingBERTModel.py Script.")
