'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute

Script that iwll run statistical tests for comparing tokenizers with
MannWhitney U-test or classifiers with the McNemar test.

Input: 2 or more .xlsx fiels of different model testing results.
output: NLPRR_ExperimentsSummary.csv containing a comparison of descriptive
statistics, and StatsTests_<stats test>.csv containing the multiple
comparison testing results of the chosen test.
'''
import argparse
import pandas as pd
import os
from tqdm import tqdm
from statsmodels.stats.multicomp import MultiComparison
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats.stats import FriedmanchisquareResult
from scipy.stats import mannwhitneyu

def mcnemar_test(a,b):
    ct = confusion_matrix(a,b)
    test = mcnemar(ct)
    return FriedmanchisquareResult(statistic=test.statistic,
                                   pvalue=test.pvalue)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str,
                    help="number of epochs of training")
parser.add_argument("--stat_test", type=str, default='MannWhitney',
                    help="Adhoc test performed. ")
opt = parser.parse_args()


print('-'*80)
print(opt)
print('-'*80)

# Building a summary spread sheet of results
filelist = []
for root, dirs, files in os.walk(opt.folder):
    for file in files:
        if file.endswith('.xlsx'):
            #append the file name to the list
            filelist.append(os.path.join(root,file))

new_df = {'FieldExtraction':[],
          'Model': [],
          'Accuracy': [],
          'G.F1': [],
          'Weighted_F1': [],
          'Weighted_precision': [],
          'Weighted_recall': []}
for file in tqdm(filelist):
    df = pd.read_excel(file, engine='openpyxl', sheet_name='Summary_Metrics')
    for fld in df.columns[1:]:
        new_df[fld].append(df[fld].values)
    new_df['FieldExtraction'].append(file.split('/')[-2])
    new_df['Model'].append(file.split('/')[-1])


new_df = pd.DataFrame(new_df)
new_df.to_csv(opt.folder +'/NLPRR_ExperimentsSummary.csv')

# Build a spread sheet with statistical testing of the results

res = [[file.split('/')[-2], file.split('/')[-1],
        pd.read_excel(file, engine='openpyxl', sheet_name='Test_results')]
       for file
       in filelist]
fields = list(set([r[0] for r in res]))

if opt.stat_test== 'MannWhitney':
    stats_tests = {'Field': [],
                   opt.stat_test + '_Acc':[],
                   opt.stat_test + '_G.F1':[],
                   'Sample_size': []}
    for fld in fields:
        df = pd.DataFrame()
        for i, r in enumerate([r for r in res if r[0]==fld]):
            df[r[1]] = r[-1]['Accuracy'].values

        stacked_data = df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'treatment',
                                                    0:'result'})
        MultiComp = MultiComparison(stacked_data['result'],
                                    stacked_data['treatment'])
        comp = MultiComp.allpairtest(mannwhitneyu, method='Holm')
        stats_tests['Field'].append(fld)
        stats_tests[opt.stat_test + '_Acc'].append(str(comp[0]))
        stats_tests['Sample_size'].append(df.shape[0])
        df = pd.DataFrame()
        for i, r in enumerate([r for r in res if r[0]==fld]):
            df[r[1]] = r[-1]['G.F1'].values

        stacked_data = df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'treatment',
                                                    0:'result'})
        MultiComp = MultiComparison(stacked_data['result'],
                                    stacked_data['treatment'])
        comp = MultiComp.allpairtest(mannwhitneyu, method='Holm')
        stats_tests[opt.stat_test + '_G.F1'].append(str(comp[0]))

if opt.stat_test== 'McNemar':
    stats_tests = {'Field': [],
                   opt.stat_test + '_ result':[],
                   'Sample_size': []}
    for fld in fields:
        df = pd.DataFrame()
        samplesets = [r for r in res if r[0]==fld]
        sizes = [x[-1].shape[0] for x in samplesets]
        if not all(x==sizes[0] for x in sizes) and len(samplesets)==2:
            all_df = [r[-1] for r in samplesets]
            all_df[1] = all_df[1][all_df[1].subject.isin(all_df[0].subject)]
        else:
            all_df = [r[-1] for r in samplesets]

        for _ in range(len(samplesets)):
            samplesets[_][-1] = all_df[_]
        for i, r in enumerate(samplesets):
            df[r[1]] = (r[-1]['PR']==r[-1]['GT']).values.astype(int)



        # Stack the data (and rename columns):

        stacked_data = df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'treatment',
                                                    0:'result'})
        MultiComp = MultiComparison(stacked_data['result'],
                                    stacked_data['treatment'])
        comp = MultiComp.allpairtest(mcnemar_test, method='Holm')
        stats_tests['Field'].append(fld)
        stats_tests[opt.stat_test + '_ result'].append(str(comp[0]))
        stats_tests['Sample_size'].append(df.shape[0])

stats_tests = pd.DataFrame(stats_tests)
stats_tests.to_csv(opt.folder +'/StatsTests_' + opt.stat_test + '.csv')