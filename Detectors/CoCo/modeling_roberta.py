# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Based on code from the above authors, modifications made by Xi'an Jiaotong University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    RobertaModel,
    RobertaForSequenceClassification,
    BertPreTrainedModel,
)
from additional_model import GCNGraphAgg

logger = logging.getLogger(__name__)


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, queue):
    """
    Update memory bank by batch window slide; hidden_batch_feats must be normalized
    """
    assert hidden_batch_feats.size()[1] == queue.size()[1]

    queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)

    return queue


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, graph_node_size=None):
        super(RobertaClassificationHead, self).__init__()
        if graph_node_size:
            self.dense = nn.Linear(
                config.hidden_size + graph_node_size, config.hidden_size
            )
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForGraphBasedSequenceClassification(
    BertPreTrainedModel
): 
    def __init__(self, config):
        config.output_hidden_states = True
        config.output_attentions = True
        super(RobertaForGraphBasedSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config, graph_node_size=None)
        self.gcn_layer = config.task_specific_params["gcn_layer"]
        self.max_node_num = config.task_specific_params["max_nodes_num"]
        self.max_sentences = config.task_specific_params["max_sentences"]
        self.max_sen_replen = config.task_specific_params["max_sen_replen"]
        self.attention_maxscore = config.task_specific_params["attention_maxscore"]
        self.relation_num = config.task_specific_params["relation_num"]

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(
            config, graph_node_size=self.max_sen_replen
        )

        self.graph_aggregation = GCNGraphAgg(
            config.hidden_size,
            self.max_sentences,
            self.gcn_layer,
            self.max_sen_replen,
            self.attention_maxscore,
            self.relation_num,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        nodes_index_mask=None,
        adj_metric=None,
        node_mask=None,
        sen2node=None,
        sentence_mask=None,
        sentence_length=None,
        batch_id=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0][:, 0, :]
        # print(f'sequence_output:{sequence_output}')
        hidden_states = outputs[2][0]
        graph_rep = self.graph_aggregation(
            hidden_states,
            nodes_index_mask,
            adj_metric,
            node_mask,
            sen2node,
            sentence_mask,
            sentence_length,
        )
        # print(f'graph_rep:{graph_rep}')
        whole_rep = torch.cat([sequence_output, graph_rep], dim=-1)

        logits = self.classifier(whole_rep, dim=-1)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs, whole_rep
