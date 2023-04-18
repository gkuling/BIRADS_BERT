from nltk.tokenize import sent_tokenize
import re
import numpy as np
from difflib import SequenceMatcher

# regex experession used to find dates in text
date_regex = '\d+ ?\/\d+\/\d+|' \
             '(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), ' \
             '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(' \
             '?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(' \
             'Nov|Dec)(?:ember)) (\d{2}|\d{1}),? (20|19)\d{2}|' \
             '(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) ' \
             '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(' \
             '?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(' \
             'Nov|Dec)(?:ember)) (\d{2}|\d{1}) (20|19)\d{2}|'\
             '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(' \
             '?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(' \
             'Nov|Dec)(?:ember)) (\d{2}|\d{1}), (20|19)\d{2}|'\
             '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(' \
             '?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(' \
             'Nov|Dec)(?:ember)) (\d{2}|\d{1}) (20|19)\d{2}'

def report_preprocess(report_text):
    """
    A set of pre processing tasks for a radiology report.
    :param report_text: orginal raw text of the file.
    :return: dict containing:
            {rep_in_para: report in paragraphs,
             sig_in_para: the radiologist signature,
             rep_in_para_redacted: a redacted version of the report in
                paragraphs,
             add_in_para: addendum in paragraphs,
             add_in_para_redacted: a redacted version of the addendum in
                paragraphs}
    """
    # Task 1: fix spacing
    text = re.sub(' +', ' ', report_text)
    text = re.sub('\?', '', text)

    # Task 2: put the report into paragraphs, and extract signatures and
    # addendums
    text_dict = parse_paragraphs(text)

    text_dict = double_check_text_dictionary(text_dict)

    # Task 3: redact specific identifiers
    if 'rep_in_para' in text_dict.keys():
        text_dict['rep_in_para_redacted'] = remove_unwanted_phrases(
            text_dict['rep_in_para'])
    if 'add_in_para' in text_dict.keys():
        text_dict['add_in_para_redacted'] = remove_unwanted_phrases(
            text_dict['add_in_para'])
    unwanted_char = ['_', '*']

    # Task 4: remove unwanted character patterns.
    for char in unwanted_char:
        for key in text_dict:
            if char in text_dict[key]:
                text_dict[key] = text_dict[key].replace(char, '')

    for key in text_dict:
        if '..' in text_dict[key]:
            text_dict[key] = text_dict[key].replace('..', '.')

    return text_dict

def easy_para_parse(line_list):
    """
    The most common way to split the document into paragraphs. Based on
    '\n\n' separations.
    :param line_list: a list of the report that has been split into lines.
    :return: str, the lines reconnected with '\n' between each paragraph
    """
    assert type(line_list)==list
    line_list = [ln.strip() for ln in line_list]
    if line_list[0]!= '':
        line_list = [''] + line_list
    lpd = [np.abs((len(line_list[i]) - len(line_list[i + 1]))) / (len(line_list[i]) + 1e-6)
           for i in range(len(line_list) - 1)]
    paragraph_beginnings = [i for i in range(len(lpd)) if lpd[i] >= 1e6]
    paragraphs = [
        ' '.join(
            line_list[paragraph_beginnings[i]:paragraph_beginnings[i + 1]]
        ).strip()
        for i in range(len(paragraph_beginnings) - 1)
    ]
    paragraphs.append(' '.join(line_list[paragraph_beginnings[-1]:]).strip())

    ret_text = append_periods_where_needed(paragraphs)
    if ':\n' in ret_text:
        if ret_text.index(':\n')-ret_text.index('\n')>0:
            ret_text = re.sub(':\n', ': ', ret_text)
        else:
            ret_text = ret_text[:re.search(':\n', ret_text).start()] + '.\n' + \
                      ret_text[re.search(':\n', ret_text).end():]
            ret_text = re.sub(':\n', ': ', ret_text)
    return ret_text

def double_check_text_dictionary(txt_dict):
    if re.search('[,A-za-z0-9]\n[a-zA-Z0-9]',txt_dict['sig_in_para']):
        for _ in range(len(
                re.findall('[,A-za-z0-9]\n[a-zA-Z0-9]', txt_dict['sig_in_para'])
        )):
            problem = re.search('[,A-za-z0-9]\n[a-zA-Z0-9]',
                                txt_dict['sig_in_para']).group()
            fix = re.search('[,A-za-z0-9]\n[a-zA-Z0-9]', txt_dict[
                'sig_in_para']).group().replace('\n', ' ')
            txt_dict['sig_in_para'] = txt_dict['sig_in_para'].replace(problem,
                                                                      fix)
    if re.search('[A-za-z]:[a-zA-Z]',txt_dict['rep_in_para']):
        for _ in range(len(
                re.findall('[A-za-z]:[a-zA-Z]', txt_dict['rep_in_para'])
        )):
            problem = re.search('[A-za-z]:[a-zA-Z]',
                                txt_dict['rep_in_para']).group()
            fix = re.search('[A-za-z]:[a-zA-Z]',
                            txt_dict['rep_in_para']).group().replace(':', ': ')
            txt_dict['rep_in_para'] = txt_dict['rep_in_para'].replace(problem,
                                                                      fix)
    if any([True if re.search('This (addendum|report) was electronically ('
                              'signed|dictated and '
                              'signed)|Dictated:|Electronically signed by',
                              para) and para!=''
            else False
            for para in txt_dict['rep_in_para'].split('\n')]):
        txt_dict['sig_in_para'] = '\n'.join(
            [para for para in txt_dict['rep_in_para'].split('\n')
             if re.search('This (addendum|report) was electronically ('
                              'signed|dictated and '
                              'signed)|Dictated:|Electronically signed by',
                              para)]
        ) + '\n' + txt_dict['sig_in_para']
        txt_dict['rep_in_para'] = '\n'.join(
            [para for para in txt_dict['rep_in_para'].split('\n')
             if not re.search('This (addendum|report) was electronically ('
                          'signed|dictated and '
                          'signed)|Dictated:|Electronically signed by',
                          para)]
        ) + '\n'

    if any([True if not re.search('This (addendum|report) was electronically ('
                              'signed|dictated and '
                              'signed)|Dictated:|Electronically signed by',
                              para) and para !='' and para != 'None'
            else False
            for para in txt_dict['sig_in_para'].split('\n')]):
        rep_addendum_indicator = \
            [True if not re.search('This (addendum|report) was electronically ('
                                   'signed|dictated and '
                                   'signed)|Dictated:|Electronically signed by',
                                   para) and para != '' else False for para in
             txt_dict['sig_in_para'].split('\n')
             ][0]
        if rep_addendum_indicator:
            txt_dict['rep_in_para'] = txt_dict['rep_in_para'] + '\n'.join(
                [para for para in txt_dict['sig_in_para'].split('\n')
                 if not re.search('This (addendum|report) was electronically ('
                              'signed|dictated and '
                              'signed)|Dictated:|Electronically signed by',
                              para)]
            ) + '\n'
            txt_dict['sig_in_para'] = '\n'.join(
                [para for para in txt_dict['sig_in_para'].split('\n')
                 if re.search('This (addendum|report) was electronically ('
                              'signed|dictated and '
                              'signed)|Dictated:|Electronically signed by',
                              para)]
            ) + '\n'
        else:
            sig_para = txt_dict['sig_in_para'].split('\n')
            add_start = [i for i in range(len(sig_para))
                         if not re.search('This (addendum|report) was '
                                          'electronically ('
                                          'signed|dictated and '
                                          'signed)|Dictated:|Electronically '
                                          'signed by',
                                          sig_para[i])][0]
            txt_dict['sig_in_para'] = '\n'.join(sig_para[:add_start]) + '\n'
            if 'Adm' in txt_dict.keys():
                txt_dict['Adm'] = '\n'.join(sig_para[add_start:]) + '\n' + \
                                  txt_dict['Adm']
            else:
                txt_dict['Adm'] = '\n'.join(sig_para[add_start:]) + '\n'

    return txt_dict

def append_periods_where_needed(para_list):
    """
    This function makes sure there is a period at the end of every paragraph.
    :param para_list: list containing paragraphs.
    :return: str
    """
    assert type(para_list)==list

    rt_txt = ''
    for par in para_list:
        if par!='':
            if re.search('.*[.?!:]$|.*\d{2}:\d{2}$', par):
                if re.search('.*\d{2}:\d{2}$', par):
                    rt_txt += par \
                                + '.\n'
                else:
                    rt_txt += par \
                                + '\n'
            else:
                if re.search('\s*$', par):
                    rt_txt += par[:re.search('\s*$', par).start()] + '.\n'
                else:
                    rt_txt += par + '.\n'
    return rt_txt

def extract_features_for_clustering(text):
    """
    Extracts the presence of specific word phrase features that are used to
    determine the overall structure of the document.
    :param text: original report in str form
    :return: feature vector
    """
    key_phrases = [
        'Electronically signed by',
        'This report was electronically signed|Dictated:',
        'This addendum was electronically signed',
        'This report was electronically dictated and signed',
        'This addendum was electronically dictated and signed',
        'Addendum|ADDENDUM|addendum',
        '--- Addendum ---',
        '\n_____________\n?',
        '\*\*\*\*\*\*\*\* ORIGINAL REPORT \*\*\*\*\*\*\*\*',
        'Report content:',
        '\*\*\*\*\*\*\*\* ADDENDUM( #\d{1})? \*\*\*\*\*\*\*\*'
    ]
    word_features = [1 if re.search(phrase, text) else 0
                     for phrase in key_phrases]
    return word_features

def parse_paragraphs(input_text):
    """
    Parses the original report into paragraphs.
    :param input_text:
    :return: dict, containing rep_in_para, and sig_in_para
    """
    input_text = '\n'.join(eliminate_duplicated_lines(input_text))
    ret_dict = {}
    doc_phrase_features = extract_features_for_clustering(input_text)

    if doc_phrase_features==[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]:
        split_doc = re.split('\n_____________\n?',input_text)
        split_doc = [sec for sec in split_doc if sec != '']
        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     not re.search('This report was electronically signed',
                                   sec)])
        )
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     re.search('This report was electronically signed',
                                   sec)])
        )
        return ret_dict

    if doc_phrase_features == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:

        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            input_text[:re.search('This report was electronically signed|Dictated:',
                                 input_text).start()]
        )
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            input_text[re.search('This report was electronically signed|Dictated:',
                                 input_text).start():]
        )
        return ret_dict

    if doc_phrase_features == [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]:
        split_doc = re.split('\n_____________\n?', input_text)
        split_doc = [sec for sec in split_doc if sec != '']
        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     not re.search('This report was electronically ('
                                   'signed|dictated and signed)|Electronically signed by',
                                   sec)])
        )
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     re.search('This report was electronically ('
                               'signed|dictated and signed)|Electronically signed by',
                               sec)])
        )
        return ret_dict

    if doc_phrase_features == [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]:
        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            input_text[:re.search('This (report|addendum) was electronically ('
                                   'signed|dictated and signed)',
                                 input_text).start()]
        )
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            input_text[re.search('This (report|addendum) was electronically ('
                                   'signed|dictated and signed)',
                                 input_text).start():]
        )
        return ret_dict

    if doc_phrase_features == [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]:

        split_doc = re.split('\n_____________\n?', input_text)
        split_doc = [sec for sec in split_doc if sec != '']
        if re.search('--- Addendum ---', input_text).start() < re.search(
                '\n_____________\n?', input_text).start():
            a = []
            for i1 in range(len(split_doc)):
                b =re.split('--- Addendum ---', split_doc[i1])
                a.append([b[i] if i==0  else '--- Addendum ---' + b[i]
                     for i in range(len(b))])
            split_doc = [item for sublist in a for item in sublist]


        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc
                     if not re.search('--- Addendum ---',
                           sec) and
                     not re.search('This report was electronically ('
                                   'signed|dictated and '
                                   'signed)|Electronically signed by',
                                   sec)])
        )
        leftover_doc = ''.join([sec for sec in split_doc if
                                re.search('--- Addendum ---', sec) or
                                re.search('This report was electronically ('
                                          'signed|dictated and '
                                          'signed)|Electronically signed by',
                                          sec)])
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            leftover_doc[:re.search('--- Addendum ---',
                                    leftover_doc).start()].strip()
        )
        ret_dict['Adm'] = split_report_into_paragraphs(
            leftover_doc[re.search('--- Addendum ---',
                                   leftover_doc).start():]
        )

        return ret_dict

    if doc_phrase_features == [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0] :
        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            input_text[:re.search('This report was electronically ('
                                  'signed|dictated and signed)|Dictated:',
                                  input_text).start()]
        )
        leftover_doc = input_text[re.search('This report was electronically ('
                                            'signed|dictated and signed)'
                                            '|Dictated:',
                                  input_text).start():]
        if re.search('Addendum|ADDENDUM|addendum', leftover_doc) and \
            re.search('Addendum|ADDENDUM|addendum', leftover_doc).start() > \
                re.search('This report was electronically (signed|dictated '
                          'and signed)|Dictated:',
                          leftover_doc).start():
            ret_dict['sig_in_para'] = split_report_into_paragraphs(
                leftover_doc[:re.search('Addendum|ADDENDUM|addendum',
                                        leftover_doc).start()]
            )
            ret_dict['Adm'] = split_report_into_paragraphs(
                leftover_doc[re.search('Addendum|ADDENDUM|addendum',
                                       leftover_doc).start():]
            )
        else:
            ret_dict['sig_in_para'] = split_report_into_paragraphs(
                leftover_doc
            )
        return ret_dict

    if doc_phrase_features == [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1] or \
            doc_phrase_features == [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1] or \
            doc_phrase_features == [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]:

        assert re.search('\*\*\*\*\*\*\*\* ORIGINAL REPORT \*\*\*\*\*\*\*\*',
                         input_text).start() > \
               re.search('--- Addendum ---|\*\*\*\*\*\*\*\* ADDENDUM( #\d{1})? \*\*\*\*\*\*\*\*',
                         input_text).start()
        original_report = input_text[re.search(
            '\*\*\*\*\*\*\*\* ORIGINAL REPORT \*\*\*\*\*\*\*\*',
            input_text).start():]
        split_doc = re.split('\n_____________\n?', original_report)
        split_doc = [sec for sec in split_doc if sec != '']
        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     not re.search('This report was electronically signed',
                                   sec)])
        )
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     re.search('This report was electronically signed',
                               sec)])
        )
        ret_dict['Adm'] = split_report_into_paragraphs(
            input_text[
            re.search('--- Addendum ---|\*\*\*\*\*\*\*\* ADDENDUM( #\d{1})? \*\*\*\*\*\*\*\*',
                      input_text).start():re.search(
                '\*\*\*\*\*\*\*\* ORIGINAL REPORT \*\*\*\*\*\*\*\*',
                input_text).start()]
        )
        ret_dict['formatter'] = split_report_into_paragraphs(
            input_text[:re.search('--- Addendum ---|\*\*\*\*\*\*\*\* ADDENDUM( #\d{1})? \*\*\*\*\*\*\*\*',
                                  input_text).start()]
        )
        return ret_dict

    if doc_phrase_features == [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]:

        split_doc = re.split('\n_____________\n?', input_text)
        split_doc = [sec for sec in split_doc if sec != '']
        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     not re.search('This report was electronically ('
                                   'signed|dictated and '
                                   'signed)|Electronically signed by',
                                   sec)])
        )
        post_rep = ''.join([sec for sec in split_doc if
                            re.search('This report was electronically ('
                                      'signed|dictated and signed)|Electronically signed by',
                                      sec)])

        split_doc = post_rep.split('\n')
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            '\n'.join([sec for sec in split_doc
                       if re.search('This report was electronically ('
                                    'signed|dictated and signed)|Electronically signed by',
                                    sec)]).strip()
        )
        ret_dict['Adm'] = split_report_into_paragraphs(
            '\n'.join([sec for sec in split_doc
                       if not re.search('This report was electronically ('
                                        'signed|dictated and signed)|Electronically signed by',
                                        sec)]).strip()
        )
        return ret_dict

    if doc_phrase_features == [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1] or \
            doc_phrase_features == [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]:

        split_doc = re.split('\n_____________\n?', input_text)
        split_doc = [sec for sec in split_doc if sec != '']
        ret_dict['rep_in_para'] = ''
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     re.search('This report was electronically signed',
                               sec)])
        )
        ret_dict['Adm'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
             re.search(
                 '--- Addendum ---|\*\*\*\*\*\*\*\* ADDENDUM( #\d{1})? \*\*\*\*\*\*\*\*',
                 sec)])
        )
        ret_dict['formatter'] = split_report_into_paragraphs(
            ''.join([sec for sec in split_doc if
                     not re.search('--- Addendum ---|\*\*\*\*\*\*\*\* ADDENDUM( #\d{1})? \*\*\*\*\*\*\*\*',
                               sec) and
                     not re.search('This report was electronically signed',
                               sec)])
        )
        return ret_dict

    if doc_phrase_features == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0] or \
            doc_phrase_features == [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0] or \
            doc_phrase_features == [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]:
        if re.search('Report content:', input_text) and \
            re.search('Report content:', input_text).start() < \
            re.search('--- Addendum ---', input_text).start():
            ret_dict['formatter'] = input_text[:re.search('Report content:',
                                                          input_text).start()]
            leftover = input_text[
                       re.search('Report content:', input_text).start():]
            split_doc = leftover.split('\n')
        else:
            split_doc = input_text.split('\n')

        ret_dict['rep_in_para'] = ''
        ret_dict['sig_in_para'] = split_report_into_paragraphs(
            '\n'.join([sec for sec in split_doc
                       if re.search('This report was electronically ('
                                    'signed|dictated and signed)',
                                    sec)])
        )
        ret_dict['Adm'] = split_report_into_paragraphs(
            '\n'.join([sec for sec in split_doc
                       if not re.search('This report was electronically ('
                                        'signed|dictated and signed)',
                                        sec)])
        )
        return ret_dict

    else:

        ret_dict['rep_in_para'] = split_report_into_paragraphs(
            input_text
        )
        ret_dict['sig_in_para'] = ''
        return ret_dict

def split_report_into_paragraphs(rp):
    """
    Splits the report into specific paragraphs.
    :param rp: str, original report
    :return:
    """
    assert type(rp)==str

    rep_lines = rp.split('\n')

    if '\n\n' in rp[:int(len(rp)/2)] and \
            not re.search('breast density (<=?|>=?):? 75%', rp.lower()):
        ### MAke sure the report doesn't have repeated line entries

        rp_n_pr = easy_para_parse(rep_lines)
    elif re.search('breast density (<=?|>=?):? 75%', rp.lower()):
        try1 = easy_para_parse(rep_lines)
        rp_n_pr = append_periods_where_needed(
            try1.replace('Breast Density',
                         '\nBreast Density').split('\n')
        )
    elif '\n' in rp[:int(len(rp)/2)] and \
            '\n\n' not in rp[:int(len(rp)/2)] and \
            len(rp.split('\n'))>2 and \
            not re.search('[a-z0-9,]\n[a-z0-9,]', rp):
        rp_n_pr = append_periods_where_needed(rp.split('\n'))
    elif '\n' in rp and re.search('[a-z0-9,]\n[a-z0-9,]', rp):
        a = re.search('[a-z0-9,]\n[a-z0-9,]', rp)
        while a:
            rp = rp.replace(rp[a.start():a.end()],
                            rp[a.start():a.end()].replace('\n', ' '))
            a = re.search('[a-z0-9,]\n[a-z0-9,]', rp)
        rp_n_pr = rp.replace('_', '')
    else:
        rp_n_pr = '\n'.join(rep_lines)
        if type(rp_n_pr)!=list:
            rp_n_pr = append_periods_where_needed([rp_n_pr])
        else:
            rp_n_pr = append_periods_where_needed(rp_n_pr)
    return rp_n_pr

def _remove_phrasing(ipt_txt, phrase, replacement):
    """
    removes the given phrase and replaces it with the replacement
    :param ipt_txt: string to change
    :param phrase: unwatned phrase to be removed
    :param replacement: replacement phrase
    :return: string with the phrase replaced with replacement
    """
    a = re.search(
        phrase,
        ipt_txt)
    it =0
    while a and it<=50:
        ipt_txt = re.sub(a.group(), replacement,
                      ipt_txt)
        a = re.search(
            phrase,
            ipt_txt)
        it+=1
        if it>50:
            raise Exception('While Loop fail. ' + str(phrase))
    return ipt_txt

def remove_unwanted_phrases(text):
    """
    Remove unwanted phrases from the report, such as time stamps, dates,
    years, months, and patient nubers that are 7 figures long
    :param text: str
    :return: str
    """
    concerns = [
        ['\d{2}: ?\d{2}:\s?\d{2}|\d{1}:\s?\d{2}|\d{2}:\s?\d{2}',
         '[TIME]'],
        [date_regex,
         '[DATE]'],
        ['(20|19)\d{2}',
         '[YEAR]'],
        ['(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun('
                  '?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|('
                  'Nov|Dec)(?:ember)) ',
         '[MONTH]'],
        ['\d{7}',
         '[PTnum]']
    ]

    for concern in concerns:
        text = _remove_phrasing(text,
                                concern[0],
                                concern[1])
    return text

def eliminate_duplicated_lines(txt):
    """
    If there are duplicated lines in the report, this function can be used to
    correct this.
    :param txt: str
    :return: list of lines with repetition.
    """
    assert type(txt)==str
    rep_lines = txt.split('\n')
    res = [rep_lines[0]]
    for ln in rep_lines:
        if ln == '':
            res.append(ln)
        elif ln != res[-1]:
            res.append(ln)
    return res

def doubled_label_fix(lbls, num, previous):
    """
    Fix ground truth mix up where a sentence belogns to two sections in a
    report.
    :param lbls: The two labels that are confusing.
    :param num: The sentence number in the report
    :param previous: previous sentence label
    :return: The correct label to use
    """
    if (lbls == ['Title', 'Findings'] or lbls == ['Title', 'Impression']) and \
            num <= 2:
        lbls = ['Title']
    if (lbls == ['Title', 'Findings'] or lbls == ['Title', 'Impression']) and \
            num >= 2:
        lbls = ['Findings']
    if lbls == ['Findings', 'Impression']:
        lbls = [previous]
    if 'HX' in lbls and num<5:
        lbls= ['HX']
    if 'HX' in lbls and num > 5:
        lbls = ['Dx']
    if lbls == ['PrIM', 'Findings'] and previous==['HX']:
        lbls = ['PrIM']
    return lbls

def check_if_line_in_section(ln, section, sec_of_interest):
    """
    Check if a given line is contained in a paragraph.
    :param ln: str of a sentence
    :param section: str of multiple sentences
    :param sec_of_interest: Section you're investigating
    :return: binary, True if ln is in section, False if not
    """
    if sec_of_interest=='Dx':
        if section.find(ln)>-1:
            return True
        elif re.search('bi-?rads', ln.lower()) and \
                any(char.isdigit() for char in ln):
            return True
        else:
            return False
    elif sec_of_interest=='Impression':
        if section.find(ln)>-1:
            return True
        elif len(ln)<len(section):
            return any([
                SequenceMatcher(None,
                                ln,
                                section[i:i + len(ln)]
                                ).quick_ratio()
                     >= .95
                     for i in range(0, len(section) - len(ln), 5)]
                )
        else:
            return False
    elif sec_of_interest=='Title':
        if section.find(ln)>-1:
            return True
        elif SequenceMatcher(None, ln, section).ratio()>0.75:
            return True
        else:
            return False
    elif sec_of_interest=='Sig':
        if section.find(ln)>-1:
            return True
        elif SequenceMatcher(None, ln, section).ratio()>0.75:
            return True
        else:
            return False
    elif sec_of_interest=='Findings':
        if section.find(ln)>-1:
            return True
        elif len(ln)<len(section):
            return any([
                SequenceMatcher(None,
                                ln,
                                section[i:i + len(ln)]
                                ).quick_ratio()
                     >= .95
                     for i in range(0, len(section) - len(ln), 5)]
                )
        else:
            return False
    elif sec_of_interest=='Procedure':
        if section.find(ln)>-1:
            return True
        elif len(ln)<len(section):
            return any([
                SequenceMatcher(None,
                                ln,
                                section[i:i + len(ln)]
                                ).quick_ratio()
                     >= .95
                     for i in range(0, len(section) - len(ln), 5)]
                )
        else:
            return False
    else:
        if section.find(ln)>-1:
            return True
        elif len(ln)<len(section):
            return any([
                SequenceMatcher(None,
                                ln,
                                section[i:i + len(ln)]
                                ).quick_ratio()
                     >= .95
                     for i in range(0, len(section) - len(ln), 5)]
                )
        else:
            return SequenceMatcher(None,
                                   ln,
                                   section).ratio()>0.95

def determine_report_GT(orig_sents, sectionized, secs):
    """
    Determine the section label for each sentence in orig_sents,
    with sectionized (dict) containing GT sections.
    :param orig_sents:, list of sents
    :param sectionized: dict, GT sectionized report
    :param secs: sections to be evaluated.
    :return: list of labels for orig_sents
    """
    eval = []
    for lin_num, snt in enumerate(orig_sents):
        lbl = [key for key in sectionized.keys() if key in secs and
               check_if_line_in_section(snt[:-1],
                                        sectionized[key],
                                        key)]
        if len(lbl) > 1:
            if lin_num==0:
                lbl = doubled_label_fix(lbl, lin_num, ['None'])
            else:
                lbl = doubled_label_fix(lbl, lin_num, eval[-1])
        eval.append(lbl[0])
    return eval

def get_sents_and_redacted(preprocessed):
    """
    Sentence tokenizes the original report and the
    redacted report.
    :param preprocessed: dict, preprocessed report data
    :return:2 lists: sents, orig_sents
    """
    sents = sent_tokenize(preprocessed['rep_in_para_redacted'])
    orig_sents = sent_tokenize(preprocessed['rep_in_para'].replace(
        '\n', ' '))
    if len(sents) != len(orig_sents):
        update = []
        cnt = 0
        for i, line in enumerate(orig_sents):
            if np.abs(len(line) - len(sents[i + cnt])) > 10.:
                update.extend([e + '.' for e in line.split('.') if e])
                cnt += 1
            else:
                update.append(line)
        orig_sents = update
    sents = [snt for snt in sents if snt != '.']
    sents = [snt.replace(' : ', ': ') for snt in sents]
    orig_sents = [snt for snt in orig_sents if snt != '.']
    orig_sents = [snt.replace(' : ', ': ') for snt in orig_sents]
    sents = join_list_item_to_content(sents)
    orig_sents = join_list_item_to_content(orig_sents)

    return sents, orig_sents

def join_list_item_to_content(snts):
    """
    Evaluates a list of sentences to determininne if an itemized list is
    present and then puts the number and list time into the same element of
    the list.
    :param snts: list of report sentences.
    :return: List of sentences where list item and number are the same lement
    of the list
    """
    if any([True if len(snt) == 2 and '.' in snt
                    and any([chr for chr in snt if
                             chr.isdigit()]) \
                    else False for snt \
            in snts]):
        ind_list_items = [i for i, ln in enumerate(snts) if
                          re.search('.*\d{1}\.+$', ln) and
                          not re.search('BI-?RADS', ln) and
                          not re.search('.*\d{4}\.+$', ln) and
                          not re.search('.*(:|/)\d{2}\.+$', ln)]
        cnt = 0
        for item in ind_list_items:
            snts[item-cnt:item-cnt+2] = [' '.join(snts[item-cnt:item-cnt+2])]
            cnt += 1
    return snts

def gt_preprocessing(txt):
    """
    Preprocessing method for the GT of section segmentation.
    :param txt: preprocessing of text in the GT data.
    :return: text that is cleaned up.
    """
    assert type(txt) == str
    txt = re.sub(' +', ' ', txt)
    txt = re.sub('\*', '', txt)
    txt = re.sub('\?', '', txt)
    if re.search('[A-Za-z]:[A-Za-z]', txt):
        txt = txt.replace(
            re.search('[A-Za-z]:[A-Za-z]',
                      txt).group(),
            re.search('[A-Za-z]:[A-Za-z]',
                      txt).group().replace(':', ': ')
        )
    if re.search('[A-Za-z]\( ', txt):
        txt = txt.replace(
            re.search('[A-Za-z]\( ',
                      txt).group(),
            re.search('[A-Za-z]\( ',
                      txt).group().replace('( ', ' (')
        )
    if re.search('ensity(>|<|>=|<=)', txt):
        txt = txt.replace(
            re.search('ensity',
                      txt).group(),
            re.search('ensity',
                      txt).group().replace('ensity', 'ensity ')
        )
    return txt
