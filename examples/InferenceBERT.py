'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py

Description: This Script gives an example of how to run the BERT model
inference on new data, either as a folder of txt file, a csv file or an xlsx
file.

Input: location of the data, a pretrained model, and the input text column.
Output: The result will be saved in the same folder as the input file with the
name extension _BERTApraised
'''
import sys
sys.path.append('.')

import argparse
import os
from utils import TrainingUtils as cutl
from models.BERTFieldExtractorWoutST import BERTFieldExtractorWoutST
import pandas as pd

cutl.set_seed(20210429)

parser = argparse.ArgumentParser()

parser.add_argument("--data_location", type=str,
                    help="folder to find experimental data. Must have a train"
                         " and test folder.")
parser.add_argument('--pre_trained_model', type=str,
                    default='bert-base-uncased',
                    help="The directory of pretrained model or the pretrained "
                         "weights from huggingface.")
parser.add_argument("--text_column", type=str, default='original_report',
                    help="The name of the column that you wish to run the "
                         "classifier on. ")
opt = parser.parse_args()

print('-'*80)
print(opt)
print('-'*80)

labeler = BERTFieldExtractorWoutST()
labeler.from_pretrained(opt.pre_trained_model)

# Begin Testing
data = cutl.load_all_data(opt.data_location)

test_res = []
incorrect = []
correct_ex = []
acc = []
ct = 0
print('Amount of Test Subjects: ' + str(len(data)))

for report in data:
    data['prediction'].append(labeler.predict(x=report[opt.text_column]))


if os.path.isdir(opt.data_location):
    pd.DataFrame(data).to_csv(os.path.join(opt.data_location,
                                           'BERTApraised.csv'))
elif opt.data_location.endswith('.xlsx'):
    pd.DataFrame(data).to_excel(
        opt.data_location.replace('.xlsx',
                                  '_BERTAppraised.xlsx'),
        engine='openpyxl'
    )
elif opt.data_location.endswith('.csv'):
    pd.DataFrame(data).to_csv(
        opt.data_location.replace('.csv',
                                  '_BERTAppraised.csv')
    )

