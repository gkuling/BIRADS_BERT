'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
import sys
sys.path.append('.')

import torch
from transformers import PretrainedConfig, \
    BertTokenizerFast, AdamW
from .BertForSequenceClassification_aux import BertForSequenceClassification_aux
from utils.TrainingUtils import training_collator
from sklearn.model_selection import train_test_split
from models.ClassificationTextDataset import ClasificationTextDataset
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import os
import numpy as np

class BERTFineTuningDeployment(object):
    """
    Object used to fine tune some bert embeddings in a downstream task.
    """
    def __init__(self, redacted_input=False, aux_data=0):
        """
        Initilization of object
        :param redacted_input: bool
        :param aux_data: int, length of auxiliary data vector.
        """
        self.redacted_input = redacted_input # choice to perform inference on
                                             # redacted data
        self.aux_data = aux_data # 0==No Extra input, >0==Extra data
                                 # considered in classifier
        self.max_len = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def setup_data_for_training(self, dt):
        '''
        A function that will organize data and reconfigure the model to have
        proper labels and outputs.
        1. First reconfigure config and model suit  your task
        2. align data into list of [Label, text features, aux_data]
        :param dt: raw data in any form you want
        :return: processed data set into a list of examples .
        '''
        raise NotImplementedError()

    def predict(self, x):
        """
        method that will implement a prediction on x
        :param x: input value
        :return: label from classifier
        """
        raise NotImplementedError()

    def from_pretrained(self, model_folder):
        """
        load model from model_folder, or a huggging face online available
        pretrained embedding.
        :param model_folder:
        :return:
        """
        self.config = PretrainedConfig.from_pretrained(model_folder)
        self.config.aux_data_size = self.aux_data

        self.model = BertForSequenceClassification_aux.from_pretrained(
            model_folder, config=self.config)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_folder)
        if self.max_len:
            self.tokenizer.model_max_length = self.max_len
        if self.tokenizer.model_max_length>512:
            self.tokenizer.model_max_length = 512
        self.model.to(self.device)

    def organize_and_prepare_data(self, data, btch_sz, portion):
        """
        organize and prepare the data for fine tuning phase.
        :param data: list, data set
        :param btch_sz: batch size
        :param portion: portion of data to train with. <=1.0
        :return:
        """
        prepared_data = self.setup_data_for_training(data)
        if type(prepared_data)==dict:
            train_data = prepared_data['training']
            dev_data = prepared_data['val']
        else:
            train_data, dev_data = train_test_split(
                prepared_data,
                stratify=[ex[0] for ex in prepared_data],
                test_size=0.15,
                random_state=20210330
            )

        if portion<1.0:
            _, train_data = train_test_split(
                train_data,
                stratify=[ex[0] for ex in train_data],
                test_size=portion,
                random_state=20210330
            )
            _, dev_data = train_test_split(
                dev_data,
                stratify=[ex[0] for ex in dev_data],
                test_size=portion,
                random_state=20210330
            )
        if self.aux_data==0:
            tr_dset = ClasificationTextDataset(train_data, aux_data=False)
            dv_dset = ClasificationTextDataset(dev_data, aux_data=False)
            tr_dtldr = DataLoader(tr_dset, btch_sz, shuffle=True,
                                  collate_fn=lambda batch: training_collator(
                                      batch))
            dv_dtldr = DataLoader(dv_dset, btch_sz, shuffle=False,
                                  collate_fn=lambda batch: training_collator(
                                      batch))
        else:
            tr_dset = ClasificationTextDataset(train_data, aux_data=True)
            dv_dset = ClasificationTextDataset(dev_data, aux_data=True)


            tr_dtldr = DataLoader(tr_dset, btch_sz, shuffle=True,
                                  collate_fn=lambda batch: training_collator(
                                      batch),
                                  drop_last=True)
            dv_dtldr = DataLoader(dv_dset, btch_sz, shuffle=False,
                                  collate_fn=lambda batch: training_collator(
                                      batch),
                                  drop_last=True)
        return tr_dtldr, dv_dtldr

    def fine_tune(self, data_set, max_epochs=4, batch_size=32,
                  ablation_portion=1.0):
        """
        fine tuning process
        :param data_set: list, dataset to be processed for training.
        :param max_epochs: max amount of epochs
        :param batch_size: batch size
        :param ablation_portion: portion of training data to use
        """
        train_dtldr, dev_dtldr = self.organize_and_prepare_data(data_set,
                                                                batch_size,
                                                                ablation_portion
                                                                )

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5,
                          eps=1e-8)
        if torch.cuda.device_count()>1:
            self.model = torch.nn.DataParallel(self.model)
        tr_loss = 0.0
        global_step = 0
        for epch in range(max_epochs):
            print(' '*5 +'Running Epoch ' + str(epch))
            epoch_iterator = tqdm(train_dtldr, desc="Iteration",
                                  position=0, leave=True)
            epoch_iterator.set_postfix({'loss': 'Initialized'})
            for batch in (epoch_iterator):
                self.model.train()
                if self.aux_data==0:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        labels=batch['labels'].to(self.device))
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        labels=batch['labels'].to(self.device),
                        aux_data=batch['aux_data'].to(self.device))
                loss = outputs['loss']
                if torch.cuda.device_count()>1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                epoch_iterator.set_postfix({'loss': loss.item()})

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               1.0)
                optimizer.step()
                self.model.zero_grad()
                global_step += 1

            epoch_iterator = tqdm(dev_dtldr, desc="Validation",
                                  position=0, leave=True)
            epoch_iterator.set_postfix({'loss': 'Initialized'})
            vl_loss = 0.0
            global_step = 0
            for batch in (epoch_iterator):
                self.model.eval()
                if self.aux_data==0:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        labels=batch['labels'].to(self.device))
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        labels=batch['labels'].to(self.device),
                        aux_data=batch['aux_data'].to(self.device))
                loss = outputs['loss']
                if torch.cuda.device_count()>1:
                    loss = loss.mean()
                vl_loss += loss.item()
                epoch_iterator.set_postfix({'loss': loss.item()})
                global_step += 1
            print('Validation Loss for Epoch ' +
                  str(epch) + ': ' + str(vl_loss/global_step))

    def save_model(self, output_dir):
        """
        Save model in a given directory
        :param output_dir: directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module \
            if hasattr(self.model, 'module') \
            else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)