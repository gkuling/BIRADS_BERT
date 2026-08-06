'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute

Shared scaffolding for the fine-tuning experiment scripts in examples/.
TrainingUtils holds training primitives (data loading, collation, seeds,
metrics); this module holds the experiment-level plumbing those scripts
share: the common argument parser, fold construction, classifier
configuration, and the results workbook writer.
'''
import argparse

from sklearn.model_selection import StratifiedKFold

from utils.TrainingUtils import dfs_tabs, split_train_test


def build_common_parser():
    """
    Parser pre-loaded with the arguments every fine-tuning script shares.
    Scripts add their own arguments before calling parse_args().
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_location", type=str,
                        help="Labeled data to train on: a folder of .txt "
                             "files that each hold a saved dict, or a single "
                             ".csv or .xlsx file.")
    parser.add_argument("--sfolder", type=str,
                        help="folder to save data")
    parser.add_argument('--pre_trained_model', type=str,
                        default='bert-base-uncased',
                        help="The directory of pretrained model or the "
                             "pretrained weights from huggingface.")
    parser.add_argument("--n_epochs", type=int, default=4,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--k_fold", action='store_true',
                        help="Choice to run the experiment in a 5 kfold "
                             "validation experiment. ")
    parser.add_argument("--use_redacted", action='store_true',
                        help="Train and predict on the redacted version of "
                             "the reports.")
    return parser


def build_folds(data, strat_column, k_fold=False):
    """
    Split data into train/test folds stratified on one column: five
    StratifiedKFold folds when k_fold is set, otherwise a single
    train/test split.
    :param data: list of dicts, one per report
    :param strat_column: str, column whose values stratify the split
    :param k_fold: bool, run 5-fold cross validation
    :return: (folds, labels) - folds is a list of {'train': [...],
        'test': [...]} dicts, labels the sorted unique values of
        strat_column
    """
    labels = list(set([row[strat_column] for row in data]))
    labels.sort()
    strat = [labels.index(row[strat_column]) for row in data]
    if k_fold:
        folds = []
        skf = StratifiedKFold(n_splits=5)

        for train_index, test_index in skf.split(data, strat):
            train, test = [data[i] for i in train_index], \
                          [data[i] for i in test_index]
            folds.append(
                {'train': train, 'test': test}
            )
        print('Beginning K-Fold training Experiment')
    else:
        train, test = split_train_test(data, strat)
        folds = [{'train': train, 'test': test}]
    return folds, labels


def configure_field_extractor(labeler, labels, field_name,
                              report_section=None):
    """
    Point a freshly loaded field extractor's config at the experiment's
    label set and field. report_section is only recorded for the
    with-segmentation arm; the without-segmentation arm's config
    deliberately carries no report_section key.
    :param labeler: a BERTFieldExtractorBase subclass instance, after
        from_pretrained
    :param labels: list, sorted label values for the field
    :param field_name: str, the field being extracted
    :param report_section: str or None, section fed to the classifier
    """
    labeler.config.num_labels = len(labels)
    labeler.config.label2id = {lbl: i for i, lbl in enumerate(labels)}
    labeler.config.id2label = {i: lbl for i, lbl in enumerate(labels)}
    extraction_config = {'field_name': field_name}
    if report_section is not None:
        extraction_config['report_section'] = report_section
    labeler.config.field_extraction_config = extraction_config


def write_experiment_results(summary_metrics, test_results, sfolder):
    """
    Write the experiment workbook next to the saved model.
    The layout is a consumed contract: post_ExperimentSummary.py locates
    result files by the '_test_results.xlsx' suffix and reads the
    'Summary_Metrics' and 'Test_results' sheets by name (and, for the
    McNemar branch, the PR/GT/subject columns within Test_results). Do
    not change any of these without updating that script.
    :param summary_metrics: DataFrame of summary metrics
    :param test_results: DataFrame of per-row test results
    :param sfolder: str, the experiment's --sfolder value
    """
    dfs_tabs([summary_metrics, test_results],
             ['Summary_Metrics', 'Test_results'],
             sfolder + '_test_results.xlsx')
