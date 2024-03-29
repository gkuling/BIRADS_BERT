# BI-RADS BERT

Implementation of BI-RADS-BERT & The Advantages of Section Segmentation. 

This implementation could be used on other radiology in house corpus as well. Labelling your own data should take the same form as reports and dataframes in './mockdata'. 

## Conda Environment setup

This project was developed using conda environments. To build the conda environment use the line of code below from the command line

```angular2html
conda create --name NLPenv --file requirements.txt --channel default --channel conda-forge --channel huggingface --channel pytorch
```


## Dataset Organization

Two datasets are needed to build BERT embeddings and fine tuned Field Extractors. 1. dataframe of SQL data, 2. labeled data for field extraction. 

Dataframe of SQL data: example file './mock_data/sql_dataframe.csv'. 
This file was efficiently made by producing a spreadsheet of all entries in the sql table and saving them as a csv file. It will require that each line of the report be split and coordinated with a SequenceNumber column to combine all the reports. Then continue to the **'How to Run BERT Pretraining'** Section.

Labeled data for Field Extraction and Section Segmentation: This data can be in one of 3 forms 
1. an xlsx file: One column must be "ReportText" where the example is present. Each column is the fields of interest and each Section of the report is a column of its own. Example of this is  
```angular2html
\mock_data\labeled_data.xlsx
```  
2. an csv file: One column must be "ReportText" where the example is present. Each column is the fields of interest and each Section of the report is a column of its own. Example of this is  
```angular2html
\mock_data\labeled_data.csv
```
3. a folder of txt files that each contain a dictionary of data. Example of txt files in './mock_data/labaled_data'. Each txt file is a save dict object with fields: 
```angular2html
example = {
    'original_report': original text report unprocessed from the exam_dataframe.csv, 
    'sectionized': dict example of the report in sections, ex. {'Title': '...', 'Hx': '...', ...}
    'PID': patient identification number,
    'date': date of the exam,
    'field_name1': name of a field you wish to classify, vlaue is the label, 
    'field_name2': more labeled fields are an option, 
    ...
}
```

## How to Run BERT Pretraining 

### Step 1: SQLtoDataFrame.py

This script can be ran to convert SQL data from a hospital records system to a dataframe for all exams. 
Hospital records keep each individual report line as a separate SQL entry, so by using 'SequenceNumber' we can assemble them in order. 

```angular2html
python ./examples/SQLtoDataFrame.py 
--input_sql ./mock_data/sql_dataframe.csv 
--save_name /folder/to/save/exam_dataframe/save_file.csv
```

This will output an 'exam_dataframe.csv' file that can be used in the next step. 

### Step 2: TextPreProcessingBERTModel.py

This script is ran to convert the exam_dataframe.csv file into a pre_training text file for training and validation, with a vocabulary size. An example of the output can be found in './mock_data/pre_training_data'.

```angular2html
python ./examples/TextPreProcessingBERTModel.py 
--dfolder ./mock_data/exam_dataframe.csv 
--ft_folder ./mock_data/pre_training_data
```

### Step 3: MLM_Training_transformers.py

This script will now run the BERT pre training with masked language modeling. The Output directory (```--output_dir```)  used is required to be empty; eitherwise the parser parameter ```--overwrite_output_dir``` is required to overwrite the files in the output directory. 
```angular2html
python ./examples/MLM_Training_transformers.py 
--train_data_file ./mock_data/pre_training_data/VocabOf39_PreTraining_training.txt 
--output_dir /folder/to/save/bert/model
--do_eval 
--eval_data_file ./mock_data/pre_training_data/PreTraining_validation.txt 
```

## How to Run BERT Fine Tuning

```--pre_trained_model``` parsed arugment that can be used for all the follwing scripts to load a pre trained embedding. The default is ```bert-base-uncased```. To get BioClinical BERT use ```--pre_trained_model emilyalsentzer/Bio_ClinicalBERT```. 

### Step 4: BERTFineTuningSectionSegmentation.py

This script will run fine tuning to train a section tokenizer with the option of using auxiliary data. 

```angular2html
python ./examples/BERTFineTuningSectionSegmentation.py 
--data_location ./mock_data/labeled_data.xlsx
--sfolder /folder/to/save/section_tokenizer
```

Optional parser arguements: 

```--aux_data``` If used then the Section Tokenizer will be trained with the auxilliary data.

```--k_fold``` If used then the experiment is run with a 5 fold cross validation. 

### Step 5: BERTFineTuningFieldExtractionWoutSectionization.py

This script will run fine tuning training of field extraction without section segmentation. 

```angular2html
python ./examples/BERTFineTuningFieldExtractionWoutSegmentation.py 
--data_location ./mock_data/labeled_data.xlsx
--sfolder /folder/to/save/field_extractor_WoutST
--field_name Modality
```

field_name is a required parsed arguement.

Optional parser arguements:

```--k_fold``` If used then the experiment is run with a 5 fold cross validation.

### Step 6: BERTFineTuningFieldExtraction.py

This script will run fine tuning training of field extraction with section segmentation.

```angular2html
python ./examples/BERTFineTuningFieldExtraction.py 
--data_location ./mock_data/labeled_data.xlsx
--sfolder /folder/to/save/field_extractor
--field_name Modality
--report_section Title
```

field_name and report_section is a required parsed arguement.

Optional parser arguements:

```--k_fold``` If used then the experiment is run with a 5 fold cross validation.

## Additional Codes 

### InferenceBERT.py

This Script gives an example of how to run the BERT model inference on new data, either as a folder of txt file, a csv file or an xlsx file.

```angular2html
python ./examples/InferenceBERT.py 
--data_location ./mock_data/input_data.xlsx
--pre_trained_model /folder/where/field_extractor/saved
--text_column ReportTxt
```

### post_ExperimentSummary.py

This code can be used to run statistical analysis of test results that are produced from BERTFineTuning codes. 

To determine the best final model, we performed statistical significance testing with a 95% confidence. We used the Mann-Whitney U test to compare the medians of different section tokenizers as the distribution of accuracy and G.F1 performance is skewed to the left (medians closer to 100%). For the field extraction classifiers, we used the McNemar test to compare the agreement between two classifiers. The McNemar test was chosen because it has been robustly proven to have an acceptable probability of Type I errors (not detecting a difference between two classifiers when there is a difference). After evaluating both configurations of field extraction explored in this paper, we performed another McNemar test to assist in choosing the best technique. All statistical tests were performed with p-value adjustments for multiple comparisons testing with Bonferonni correction.

Note: input folder must contain 2 or more .xlsx files of experiemtnal results to perform a statistical test. 
```angular2html
python ./examples/post_ExperimentSummary.py --folder /folder/where/xlsx/files/are/located --stat_test MannWhitney
```

```--stat_test``` options: 'MannWhitney' and 'McNemar'. 

'MannWhitney': MannWhitney U-Test. This test was used for the Section Tokenizer experimental results comparing the results from different models. https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test

'McNemar' : McNemar's test. This test was used for the Field Extraction experimental results comparing the results from different models. https://en.wikipedia.org/wiki/McNemar%27s_test 
## Contact 

Please post a Github issue if you have any questions.