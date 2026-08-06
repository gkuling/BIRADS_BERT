'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py

Description: Training code used to train a model on the identification of a
field in the input text. This is meant to train a classifier WITHOUT section
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

from models.BERTFieldExtractorWoutST import BERTFieldExtractorWoutST
from utils import ExperimentUtils as exutl
from utils import TrainingUtils as cutl

cutl.set_seed(20210429)

parser = exutl.build_common_parser()
parser.add_argument('--field_name', type=str, required=True,
                    help='Field of interest to be extracted from reports.')
opt = parser.parse_args()

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
    labeler = BERTFieldExtractorWoutST(redacted_input=opt.use_redacted)
    labeler.from_pretrained(opt.pre_trained_model)
    exutl.configure_field_extractor(labeler, Labels, opt.field_name)

    # Begin Training
    labeler.fine_tune(fold['train'],
                      max_epochs=opt.n_epochs,
                      batch_size=opt.batch_size)
    labeler.save_model(opt.sfolder + str(k))

    labeler = BERTFieldExtractorWoutST(redacted_input=opt.use_redacted)
    labeler.from_pretrained(opt.sfolder + str(k))

    # Begin Testing
    # Predictions are pooled across folds into per-subject PR/GT rows:
    # post_ExperimentSummary.py's McNemar branch consumes exactly this shape.
    print('Amount of Test Subjects: ' + str(len(fold['test'])))
    for report in fold['test']:
        results['subject'].append(report['filename'])
        results['GT'].append(report[opt.field_name])
        results['input'].append(report['original_report'])
        results['PR'].append(labeler.predict(x=report['original_report']))

# Summarize Testing Results
metrics = cutl.calculate_testing_metrics(results['PR'], results['GT'],
                                         cutl.new_metrics_dict())

test_results = pd.DataFrame(results)
summary_metrics = pd.DataFrame(metrics)

print('Accuracy of BERTFieldExtractorWoutST(): ' + str(np.round(
    metrics['Accuracy'][0]*100)))
print('G.F1 of BERTFieldExtractorWoutST(): ' + str(np.round(
    metrics['G.F1'][0]*100)))
print('-' * 60)

exutl.write_experiment_results(summary_metrics, test_results, opt.sfolder)
print('End of BERTFineTuningFieldExtractionWoutSegmentation.py Script')
