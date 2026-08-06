'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py

Description: Training code used to train a model on the identification of a
field in the input text. This is meant to train a classifier WITH section
segmentation as a pre processing step.

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

from models.BERTFieldExtractor import BERTFieldExtractor
from utils import ExperimentUtils as exutl
from utils import TrainingUtils as cutl

cutl.set_seed(20210429)

parser = exutl.build_common_parser()
parser.add_argument('--field_name', type=str, required=True,
                    help='Field of interest to be extracted from reports.')
parser.add_argument('--report_section', type=str, required=True,
                    help='Report Section used to extract field of interest to '
                         'be extracted from reports.')
parser.add_argument('--max_sent_len', type=int, default=512,
                    help='Length of sequence entered into the model. ')
opt = parser.parse_args()

opt.sfolder = opt.sfolder + '_ml' + str(opt.max_sent_len)

print('-'*80)
print(opt)
print('-'*80)

# Set up data for training and testing
data = cutl.load_all_data(opt.data_location)
folds, Labels = exutl.build_folds(data, opt.field_name, k_fold=opt.k_fold)

# Begin Experiment
results = {'subject': [],
           'input': [],
           'GT': [],
           'PR': []}
for k, fold in enumerate(folds):
    print('Running the ' + str(k+1) + ' kfold experiment.')
    # Initialize Model
    labeler = BERTFieldExtractor(redacted_input=opt.use_redacted,
                                 max_len=opt.max_sent_len)
    labeler.from_pretrained(opt.pre_trained_model)
    exutl.configure_field_extractor(labeler, Labels, opt.field_name,
                                    report_section=opt.report_section)

    # Begin Training
    labeler.fine_tune(fold['train'],
                      max_epochs=opt.n_epochs,
                      batch_size=opt.batch_size)
    labeler.save_model(opt.sfolder + str(k))

    labeler = BERTFieldExtractor(redacted_input=opt.use_redacted,
                                 max_len=opt.max_sent_len)
    labeler.from_pretrained(opt.sfolder + str(k))

    # Begin Testing
    # Predictions are pooled across folds into per-subject PR/GT rows:
    # post_ExperimentSummary.py's McNemar branch consumes exactly this shape.
    print('Amount of Test Subjects: ' + str(len(fold['test'])))
    for report in fold['test']:
        if 'sectionized' in report.keys():
            if opt.report_section in report['sectionized'].keys():
                # if the required report section is in the sectionized ground
                # truth, use this example. Otherwise, don't use it
                results['subject'].append(report['filename'])
                results['GT'].append(report[opt.field_name])
                results['input'].append(report['sectionized'][opt.report_section])
                results['PR'].append(
                    labeler.predict(x=report['sectionized'][opt.report_section])
                )
        elif opt.report_section in report.keys():
            if report[opt.report_section]!='':
                results['subject'].append(report['filename'])
                results['GT'].append(report[opt.field_name])
                results['input'].append(
                    report[opt.report_section])
                results['PR'].append(
                    labeler.predict(x=report[opt.report_section])
                )
        else:
             raise Exception('The data must contain a sectionized column or a '
                             'column matching the report_section')

# Summarize Testing Results
metrics = cutl.calculate_testing_metrics(results['PR'], results['GT'],
                                         cutl.new_metrics_dict())

test_results = pd.DataFrame(results)
summary_metrics = pd.DataFrame(metrics)

print('Accuracy of BERTFieldExtractor(): ' + str(np.round(
    metrics['Accuracy'][0]*100)))
print('G.F1 of BERTFieldExtractor(): ' + str(np.round(
    metrics['G.F1'][0]*100)))
print('-' * 60)

exutl.write_experiment_results(summary_metrics, test_results, opt.sfolder)
print('End of BERTFineTuningFieldExtraction.py Script')
