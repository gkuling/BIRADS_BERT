from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, \
    multilabel_confusion_matrix
import numpy as np
import random
import torch

rd_state = 20200709

def train_test_spliting(dataset, strat=None):
    """
    implementation of train test split ir or if not stratification is given.
    (see train_test_split() from scikit-learn documentation)
    :param dataset: data to be split
    :param strat: stratification for dataset
    :return: train, test
    """
    if strat is not None:
        train, test = train_test_split(dataset,
                                       test_size=0.2,
                                       stratify=strat,
                                       random_state=rd_state)
    else:
        train, test = train_test_split(dataset,
                                       test_size=0.2,
                                       random_state=rd_state)
    return train, test

def load_all_data(directory):
    """
    Pull in txt data files that have fine tuning examples saved as dict
    :param directory: folder where txt files are saved
    :return: a list of fine tuning data. A list of dict
    """
    full_data = []
    if os.path.isdir(directory):
        for file in [fl for fl in os.listdir(directory) if fl.endswith('.txt')]:
            f = open(directory + '/' + file, 'r')
            data = f.read()
            f.close()

            data = eval(data)
            data['filename'] = file
            full_data.append(data)
    elif directory.endswith('.xlsx'):
        full_data = pd.read_excel(directory, engine='openpyxl')
        full_data = full_data.fillna('')
        full_data = full_data.to_dict('records')
    elif directory.endswith('.csv'):
        full_data = pd.read_csv(directory)
        full_data = full_data.fillna('')
        full_data = full_data.to_dict('records')
    else:
        raise Exception("'directory' must be a folder filled with text files "
                        "or a .xlsx file. ")

    return full_data

def calculate_testing_metrics(pred, gt, current_metrics):
    """
    Implementation to update the mcurrent_metrics dict with the next test sample
    :param pred: prediction from model
    :param gt: ground truth tot est against
    :param current_metrics: dict containing all metrics as keys
    :return: current emtrics with the test results appended on
    """
    for key in current_metrics.keys():
        current_metrics[key].extend([calculate_metric(pred, gt, key)])
    return current_metrics

def calculate_metric(pr, gt, met):
    """
    Implementation of calculating a metric based on the prediction and
    ground truth
    :param pr: prediction from the model
    :param gt: ground truth to test against
    :param met: metric to be evaluated. Implementations are Accuracy,
    Weighted_F1, Weighted_precision, Weighted_recall, G. F1 measure
    :return: metric calculated
    """
    if met == 'Accuracy':
        return len([i for i in range(len(gt)) if gt[i] == pr[i]]) / len(gt)
    elif met == 'Weighted_F1':
        return f1_score(pr, gt, average='weighted', zero_division=0)
    elif met == 'Weighted_precision':
        return precision_score(pr, gt, average='weighted', zero_division=0)
    elif met == 'Weighted_recall':
        return recall_score(pr, gt, average='weighted', zero_division=0)
    elif met == 'G.F1':
        lbls = list(set(pr + gt))
        gt = [lbls.index(e) for e in gt]
        pr = [lbls.index(e) for e in pr]
        mcm = multilabel_confusion_matrix(pr, gt)
        w = [1/(np.sum([el == i for el in gt]) + 1e-2)**2 for i in range(len(
            lbls))]
        num = 0
        den = 0
        for i, m in enumerate(mcm):
            tn, fp, fn, tp = m.ravel()
            num += tp * w[i]
            den += (2 * tp + fp + fn) * w[i]
        return 2*(num/den)
    else:
        raise Exception('Metric ' + met + ' is not implemented in '
                                          'calculate_metric. ')

def dfs_tabs(df_list, sheet_list, file_name):
    """
    Combine a set of dataframes into an excel sheet, where exach dataframe is a
    sheet in the excel spreadsheet.
    :param df_list: list containing the dataframes to be combined.
    :param sheet_list: Names of the sheets for each dataframe
    :param file_name: Place where you wish to save the excel file .xlsx
    """
    writer = pd.ExcelWriter(file_name)
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0)
    writer.save()

def set_seed(sd):
    """
    Sets the seed for the experiment. Controlling numpy, python, torch and
    torch cuda.
    :param sd: number
    """
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(sd)

def training_collator(btch, max_len=None):
    """
    Combines batch files into a collated tensor for traininng. Can have a max
    sequence length that you wish to adhere to, or if None will make the max
    length the longest sequence in the batch
    :param btch: batch file recievevd from the data laoder
    :param max_len: len of sequence you wish for the batch to adhere to
    :return: batch with collated data
    """
    if max_len is None:
        max_len = np.max([len(b['input_ids']) for b in btch])
    opt = np.zeros((len(btch), max_len))

    if 'aux_data' in btch[0].keys():
        aux_sz = len(btch[0]['aux_data'])
    else:
        aux_sz = 1
    label = np.zeros((len(btch), 1))
    aux_data = np.zeros((len(btch), aux_sz))
    for i, ex in enumerate(btch):
        if len(ex['input_ids']) <= max_len:
            opt[i][:len(ex['input_ids'])] = torch.tensor(ex['input_ids'])
        else:
            opt[i] = torch.tensor(ex['input_ids'][:max_len])
        label[i] = ex['labels']
        if 'aux_data' in ex.keys():
            aux_data[i] = ex['aux_data']

    return {'input_ids': torch.tensor(opt, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'aux_data': torch.tensor(aux_data, dtype=torch.float)}
