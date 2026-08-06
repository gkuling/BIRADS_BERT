# BI-RADS BERT

BERT for breast radiology reports: contextual embeddings pre-trained on breast imaging reports, with fine-tuned classifiers for **section segmentation** and **BI-RADS field extraction**.

Official implementation of [BI-RADS BERT and Using Section Segmentation to Understand Radiology Reports](https://www.mdpi.com/2313-433X/8/5/131) (Kuling, Curpen & Martel, *Journal of Imaging* 2022, 8(5):131 — [arXiv:2110.07552](https://arxiv.org/abs/2110.07552), [DOI 10.3390/jimaging8050131](https://doi.org/10.3390/jimaging8050131)).

**Headline result:** free-text breast radiology reports are segmented sentence-by-sentence into BI-RADS lexicon sections at **98% accuracy**, and using the segmented sections improves downstream field extraction.

This implementation can be adapted to other in-house radiology corpora: label your own data in the same form as the reports and dataframes in `./mock_data`.

## Methods

To determine the best final model, we performed statistical significance testing with a 95% confidence. We used the Mann-Whitney U test to compare the medians of different section tokenizers as the distribution of accuracy and G.F1 performance is skewed to the left (medians closer to 100%). For the field extraction classifiers, we used the McNemar test to compare the agreement between two classifiers. The McNemar test was chosen because it has been robustly proven to have an acceptable probability of Type I errors (not detecting a difference between two classifiers when there is a difference). After evaluating both configurations of field extraction explored in this paper, we performed another McNemar test to assist in choosing the best technique. All statistical tests were performed with p-value adjustments for multiple comparisons testing with Bonferroni correction.

The two experiment types deliberately report metrics in different shapes to feed these tests: the section-segmentation script scores each report individually (a per-report distribution for Mann-Whitney), while the field-extraction scripts pool per-subject predictions (paired agreement for McNemar). `post_ExperimentSummary.py` consumes both.

## Repository layout

- `models/` — the classifiers: `BERTSectionTokenizer` / `BERTSectionTokenizerWithAux` (the paper's two section-segmentation arms) and `BERTFieldExtractor` / `BERTFieldExtractorWoutST` (field extraction with and without section segmentation). Each pair shares a base class (`models/*Base.py`); the subclasses hold only what genuinely differs between the arms.
- `utils/` — text preprocessing (`TextPreprocessing.py`), training primitives (`TrainingUtils.py`), and shared experiment scaffolding for the example scripts (`ExperimentUtils.py`).
- `examples/` — runnable entry-point scripts for each pipeline stage, described below. Run them from the repository root.
- `mock_data/` — synthetic example data for every stage (no real patient data).

## Conda environment setup

This project was developed using conda environments. To build the conda environment use the line of code below from the command line:

```bash
conda create --name NLPenv --file requirements.txt --channel default --channel conda-forge --channel huggingface --channel pytorch
```

Note that `requirements.txt` is a conda package spec, not a pip requirements file — `pip install -r requirements.txt` will not work. The pinned versions are the ones used for the published results (see [Reproducibility notes](#reproducibility-notes)).

## Dataset organization

Two datasets are needed to build BERT embeddings and fine-tuned field extractors: 1. a dataframe of SQL data, 2. labeled data for field extraction.

**Dataframe of SQL data:** example file `./mock_data/sql_dataframe.csv`. This file was efficiently made by producing a spreadsheet of all entries in the SQL table and saving them as a csv file. It will require that each line of the report be split and coordinated with a SequenceNumber column to combine all the reports. Then continue to the **How to run BERT pretraining** section.

**Labeled data for field extraction and section segmentation:** this data can be in one of 3 forms:

1. an xlsx file: one column must be `original_report` holding the raw report text. Each field of interest and each section of the report is a column of its own. Example: `./mock_data/labeled_data.xlsx`
2. a csv file with the same columns. Example: `./mock_data/labeled_data.csv`
3. a folder of txt files that each contain a dictionary of data. Example: `./mock_data/labeled_data`. Each txt file is a saved dict object with fields:

```text
example = {
    'original_report': original text report unprocessed from the exam_dataframe.csv,
    'sectionized': dict example of the report in sections, ex. {'Title': '...', 'Hx': '...', ...}
    'PID': patient identification number,
    'date': date of the exam,
    'field_name1': name of a field you wish to classify, value is the label,
    'field_name2': more labeled fields are an option,
    ...
}
```

## How to run BERT pretraining

### Step 1: SQLtoDataFrame.py

This script can be run to convert SQL data from a hospital records system to a dataframe for all exams. Hospital records keep each individual report line as a separate SQL entry, so by using `SequenceNumber` we can assemble them in order.

```bash
python ./examples/SQLtoDataFrame.py \
  --input_sql ./mock_data/sql_dataframe.csv \
  --save_name /folder/to/save/exam_dataframe/save_file.csv
```

This will output an `exam_dataframe.csv` file that can be used in the next step.

### Step 2: TextPreProcessingBERTModel.py

This script is run to convert the exam dataframe into pre-training text files for training and validation, with a vocabulary size. An example of the output can be found in `./mock_data/pre_training_data`.

```bash
python ./examples/TextPreProcessingBERTModel.py \
  --dfolder ./mock_data/pre_training_data \
  --ft_folder ./mock_data/labeled_data
```

`--dfolder` is a folder containing the exam dataframe csv file(s); the output text files (both original and redacted variants) are written into the same folder. `--ft_folder` is the folder of labeled fine-tuning examples, which are excluded from the pre-training set.

### Step 3: MLM_Training_transformers.py

This script will now run the BERT pre-training with masked language modeling. The output directory (`--output_dir`) is required to be empty; otherwise the parser parameter `--overwrite_output_dir` is required to overwrite the files in the output directory.

```bash
python ./examples/MLM_Training_transformers.py \
  --train_data_file ./mock_data/pre_training_data/VocabOf39_PreTraining_training.txt \
  --output_dir /folder/to/save/bert/model \
  --do_eval \
  --eval_data_file ./mock_data/pre_training_data/PreTraining_validation.txt
```

The training file's name must carry the vocabulary size as digits (as produced by Step 2, e.g. `VocabOf39_...`).

## How to run BERT fine tuning

`--pre_trained_model` is a parsed argument that can be used for all the following scripts to load a pre-trained embedding. The default is `bert-base-uncased`. To get BioClinical BERT use `--pre_trained_model emilyalsentzer/Bio_ClinicalBERT`.

Optional parser arguments shared by all three scripts:

- `--k_fold` — run the experiment with a 5-fold cross validation.
- `--use_redacted` — train and predict on the redacted version of the reports (produced by the preprocessing pipeline).

### Step 4: BERTFineTuningSectionSegmentation.py

This script will run fine tuning to train a section tokenizer with the option of using auxiliary data.

```bash
python ./examples/BERTFineTuningSectionSegmentation.py \
  --data_location ./mock_data/labeled_data.xlsx \
  --sfolder /folder/to/save/section_tokenizer
```

Optional parser arguments:

- `--aux_data` — train the section tokenizer with auxiliary data (previous sentence label, sentence position, report length).
- `--data_portion` — portion of the training data to use (for ablations).

### Step 5: BERTFineTuningFieldExtractionWoutSegmentation.py

This script will run fine tuning training of field extraction without section segmentation.

```bash
python ./examples/BERTFineTuningFieldExtractionWoutSegmentation.py \
  --data_location ./mock_data/labeled_data.xlsx \
  --sfolder /folder/to/save/field_extractor_WoutST \
  --field_name Modality
```

`--field_name` is a required parsed argument.

### Step 6: BERTFineTuningFieldExtraction.py

This script will run fine tuning training of field extraction with section segmentation.

```bash
python ./examples/BERTFineTuningFieldExtraction.py \
  --data_location ./mock_data/labeled_data.xlsx \
  --sfolder /folder/to/save/field_extractor \
  --field_name Modality \
  --report_section Title
```

`--field_name` and `--report_section` are required parsed arguments.

Optional parser argument: `--max_sent_len` (default 512) sets the sequence length fed to the model. Note that the script appends `_ml<max_sent_len>` (and the fold number) to `--sfolder`, so the example above saves to `.../field_extractor_ml512<fold>`.

## Additional codes

### InferenceBERT.py

This script gives an example of how to run the BERT model inference on new data, either as a folder of txt files, a csv file or an xlsx file.

```bash
python ./examples/InferenceBERT.py \
  --data_location ./mock_data/labeled_data.csv \
  --pre_trained_model /folder/where/field_extractor/saved \
  --text_column original_report
```

The predictions are saved beside the input as `<input name>_BERTAppraised.csv` (or `.xlsx`), e.g. `./mock_data/labeled_data_BERTAppraised.csv` for the example above.

### post_ExperimentSummary.py

This code can be used to run statistical analysis of test results that are produced from the BERTFineTuning codes (see [Methods](#methods) for how each test maps onto the experiment outputs).

Note: input folder must contain 2 or more .xlsx files of experimental results to perform a statistical test.

```bash
python ./examples/post_ExperimentSummary.py --folder /folder/where/xlsx/files/are/located --stat_test MannWhitney
```

`--stat_test` options: `MannWhitney` and `McNemar`.

- `MannWhitney`: [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test), used for the section tokenizer experimental results comparing the results from different models.
- `McNemar`: [McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test), used for the field extraction experimental results comparing the results from different models.

## Reproducibility notes

- `requirements.txt` pins the exact versions used for the published results; it is kept unchanged as the reproduction record.
- The utility and I/O code has been made tolerant of current library versions where that costs nothing (pandas `ExcelWriter`/`drop`, scipy imports, `ast.literal_eval` for data files) and remains backward-compatible with the pins.
- The training path itself is not forward-portable: `transformers.TextDataset` and `transformers.AdamW` no longer exist in current `transformers`. Use the pinned environment to train; a modern-dependency port is out of scope for this archival research repo.
- The preprocessing pipeline (Steps 1–2) runs on current library versions and reproduces the shipped `mock_data/pre_training_data` outputs byte-for-byte.

## Citation

If you use this code, please cite:

```bibtex
@article{kuling2022birads,
  author  = {Kuling, Grey and Curpen, Belinda and Martel, Anne L.},
  title   = {BI-RADS BERT and Using Section Segmentation to Understand Radiology Reports},
  journal = {Journal of Imaging},
  volume  = {8},
  number  = {5},
  pages   = {131},
  year    = {2022},
  doi     = {10.3390/jimaging8050131}
}
```

## License

MIT (see [LICENSE](LICENSE)).

## Contact

Please post a GitHub issue if you have any questions.
