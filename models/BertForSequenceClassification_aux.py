'''
Copyright (c) 2020, Martel Lab, Sunnybrook Research Institute
Codes inspired by Hugging Face Transformers package code run_mlm.py
https://github.com/huggingface/transformers/blob/master/examples/pytorch
/language-modeling/run_mlm.py
'''
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import BertPreTrainedModel, BertModel

class BertForSequenceClassification_aux(BertPreTrainedModel):
    """
    BertPreTrainedModel built with the option of using Auxiliary data in
    classification
    """
    def __init__(self, config):
        """
        Initializer of BertForSequenceClassification_aux
        :param config: transformers Bert config, with an extra parameter of
            'aux_data_size'
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size +
                                    (config.aux_data_size > 0) * 128,
                                    config.num_labels)
        if config.aux_data_size > 0:
            self.aux_encoder = nn.Sequential(
                nn.Linear(config.aux_data_size, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.Tanh()
            )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        aux_data=None):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
        `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., onfig.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is
                computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is
                computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        if aux_data is not None:
            pre_classifier = self.dropout(torch.cat(
                [pooled_output, self.aux_encoder(aux_data)],
                dim=1))
        else:
            pre_classifier = self.dropout(
                pooled_output
            )
        logits = self.classifier(
            pre_classifier
        )

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
