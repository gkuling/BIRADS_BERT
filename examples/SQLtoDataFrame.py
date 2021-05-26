import argparse
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_sql", type=str,
                    help="File location of SQL data.")
parser.add_argument("--save_name", type=str,
                    help="File location you wish to csv file.")
parser.add_argument("--session_col", type=str, default="AccNum",
                    help="Label for the column that holds information on exam "
                         "session number ")
parser.add_argument("--report_col", type=str, default="ReportTxt",
                    help="Label for the column that contains report text.")
opt = parser.parse_args()

print('-'*80)
print(opt)
print('-'*80)

def eliminate_sequence_number(sql_data,
                              accession_label='AccNum',
                              textlabel='ReportTxt'):
    """
    This function takes a csv input that has a SequenceNumber column that
    give the sequence information of report text. This will then append all
    string data in the proper sequence to merge multiple lines of the same
    session into one row of information in a dataframe.
    :param sql_data: csv file containing sql output where the report data is
        split into lines characterizeed by the sequence number. Requires a
        SequenceNumber column and an ExamDate column.
    :param accession_label: str, name of the column that contains session
    number
    :param textlabel: str, name of the column that contains report text
    :return: Dataframe of all rows of sql_data file compressed into sessions
    """
    sql_data = sql_data[[textlabel, accession_label, 'SequenceNumber',
                         'ExamDate']].replace(np.nan, '', regex=True).astype({textlabel: str})

    acc_nums = list(set(sql_data[accession_label]))
    output_data = []
    acc_nums_iterator = tqdm(acc_nums, desc="SQLtoDataframe Progress: ",
                             position=0, leave=True)
    for num in acc_nums_iterator:
        temp = sql_data.loc[sql_data[accession_label] == num]
        temp = temp.drop_duplicates()
        temp = temp.sort_values(['SequenceNumber'])
        save_value = list(temp.values[0])
        col_oi = list(temp.columns).index(textlabel)
        for row in temp.itertuples():
            if row.SequenceNumber == 0:
                continue
            else:
                if row[col_oi+1] == '(null)':
                    save_value[col_oi] += '\n'
                else:
                    save_value[col_oi] += '\n' + row[col_oi+1]
        output_data.append(save_value)
        sys.stdout.write(
            '\rFinished analyzing ' + str(len(output_data)/len(acc_nums))
        )
        sys.stdout.flush()
    op_cols = list(sql_data.columns)
    rt_df = pd.DataFrame(output_data,
                         columns=op_cols)
    rt_df = rt_df.drop('SequenceNumber', 'columns')
    return rt_df


sql_reports = pd.read_csv(opt.input_sql)

reports_processed = eliminate_sequence_number(sql_reports,
                                              accession_label=opt.session_col,
                                              textlabel=opt.report_col)
reports_processed.to_csv(opt.save_name)

print('End of SQLtoDataframe Script')
