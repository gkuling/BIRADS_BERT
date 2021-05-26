import os
import numpy as np
import pandas as pd

if __name__=='__main__':
    folder = './mock_data/labeled_data_orig'
    ofolder = os.getcwd() + '/mock_data/labeled_data'
    files = [folder + '/' + fl for fl in os.listdir(folder)]
    bpe_lbl = {
        "Marked": 0,
        "Mild": 1,
        "Minimal": 2,
        "Moderate": 3,
        "Not Stated": 4
    }
    den_lbl = {
        "<= 75%": 0,
        "Dense": 1,
        "Fatty": 2,
        "Heterogeneous": 3,
        "Not Stated": 4,
        "Scattered": 5
    }
    men_lbl = {
        "Not Stated": 0,
        "Post-Menopausal": 1,
        "Pre-Menopausal": 2
    }
    mod_lbl = {
        "BIO-MG": 0,
        "BIO-MG-MRI": 1,
        "BIO-MG-US": 2,
        "BIO-MRI": 3,
        "BIO-US": 4,
        "MG": 5,
        "MG-MRI": 6,
        "MG-US": 7,
        "MRI": 8,
        "Other": 9,
        "US": 10
    }
    prv_lbl = {
        "Negative": 0,
        "Positive": 1,
        "Suspicious": 2
    }
    pur_lbl = {
        "Diagnostic": 0,
        "Screening": 1,
        "Unknown": 2
    }
    basic_labels = {'Modality': 'US',
                    'PreviousCa': 'Negative',
                    'Density': 'Not Stated',
                    'Purpose': 'Screening',
                    'BPE': 'Not Stated',
                    'Menopausal_Status': 'Not Stated'}
    rep_type = np.random.randint(low=0, high=3, size=(50,))
    for i, rep in enumerate(rep_type):
        base = eval(open(files[rep]).read())
        base['PID'] = i
        bpe = np.random.randint(low=0, high=len(bpe_lbl.keys()))
        bpe = [name for name, val in bpe_lbl.items() if val == bpe][0]
        base['BPE'] = bpe
        den = np.random.randint(low=0, high=len(den_lbl.keys()))
        den = [name for name, val in den_lbl.items() if val == den][0]
        base['Density'] = den
        men = np.random.randint(low=0, high=len(men_lbl.keys()))
        men = [name for name, val in men_lbl.items() if val == men][0]
        base['Menopausal_Status'] = men

        mod = np.random.randint(low=0, high=len(mod_lbl.keys()))
        mod = [name for name, val in mod_lbl.items() if val == mod][0]
        base['Modality'] = mod
        prv = np.random.randint(low=0, high=len(prv_lbl.keys()))
        prv = [name for name, val in prv_lbl.items() if val == prv][0]
        base['PreviousCa'] = prv

        pur = np.random.randint(low=0, high=len(pur_lbl.keys()))
        pur = [name for name, val in pur_lbl.items() if val == pur][0]
        base['Purpose'] = pur

        with open(ofolder + '/Sectioned' + str(i) + '.txt', 'w') as f:
            print(base, file=f)

    df = {'PID': [],
          'ExamDate':[],
          'ReportTxt': []}
    files = [ofolder + '/' + fl for fl in os.listdir(ofolder)]
    for fl in files:
        rep = eval(open(fl).read())
        df['PID'].append(rep['PID'])
        df['ExamDate'].append(rep['date'])
        df['ReportTxt'].append(rep['original_report'])
    df = pd.DataFrame(df)
    df.to_csv(os.getcwd() + '/mock_data/reports_by_exam.csv')
    new_df = {'AccNum': [],
              'ExamDate':[],
              'ReportTxt': [],
              'SequenceNumber': []}
    for exam in df.itertuples():
        rows = exam.ReportTxt.split('\n')
        for i, row in enumerate(rows):
            new_df['AccNum'].append(exam.PID)
            new_df['ExamDate'].append(exam.ExamDate)
            new_df['ReportTxt'].append(row)
            new_df['SequenceNumber'].append(i)
    new_df = pd.DataFrame(new_df)
    new_df.to_csv(os.getcwd() + '/mock_data/sql_dataframe.csv')
    print('')
