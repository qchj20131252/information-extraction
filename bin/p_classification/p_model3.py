"""
This module to define a neural network
"""

import json
import os
import sys
import argparse

import paddle
import paddle.fluid as fluid


def db_lstm(data_reader, word, postag, arc_relation, arc_head, conf_dict):
    """
    Neural network structure definition: stacked bidirectional
    LSTM and max-pooling
    """
    hidden_dim = conf_dict['hidden_dim']
    depth = conf_dict['depth']
    label_dict_len = data_reader.get_dict_size('label_dict')
    word_emb_fixed = True if conf_dict['word_emb_fixed'] == "True" else False
    emb_distributed = not conf_dict['is_local']
    # 4 features
    word_param = fluid.ParamAttr(name=conf_dict['emb_name'], trainable=(not word_emb_fixed))
    pos_param = fluid.ParamAttr(name='pos_emb', trainable=(not word_emb_fixed))
    arc_relation_param = fluid.ParamAttr(name='arc_relation_emb', trainable=(not word_emb_fixed))
    arc_head_param = fluid.ParamAttr(name='arc_head_emb', trainable=(not word_emb_fixed))

    conf_dict['is_sparse'] = bool(conf_dict['is_sparse'])
    word_embedding = fluid.layers.embedding(
        input=word,
        size=[data_reader.get_dict_size('wordemb_dict'), conf_dict['word_dim']],
        dtype='float32',
        is_distributed=emb_distributed,
        is_sparse=conf_dict['is_sparse'],
        param_attr=word_param)

    postag_embedding = fluid.layers.embedding(
        input=postag,
        size=[data_reader.get_dict_size('postag_dict'), 32],
        dtype='float32',
        is_distributed=emb_distributed,
        is_sparse=conf_dict['is_sparse'],
        param_attr=pos_param)

    arc_relation_embedding = fluid.layers.embedding(
        input=arc_relation,
        size=[data_reader.get_dict_size('arc_relation_dict'), 32],
        dtype='float32',
        is_distributed=emb_distributed,
        is_sparse=conf_dict['is_sparse'],
        param_attr=arc_relation_param)

    arc_head_embedding = fluid.layers.embedding(
        input=arc_head,
        size=[data_reader.get_dict_size('arc_relation_dict'), 32],
        dtype='float32',
        is_distributed=emb_distributed,
        is_sparse=conf_dict['is_sparse'],
        param_attr=arc_head_param)

    # embedding
    emb_layers = [word_embedding, postag_embedding, arc_relation_embedding, arc_head_embedding]

    # input hidden
    hidden_0_layers = [fluid.layers.fc(input=emb, size=hidden_dim, act='tanh') for emb in emb_layers]

    hidden_0 = fluid.layers.sums(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    # max-pooling
    fc_last = fluid.layers.sequence_pool(input=input_tmp[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=input_tmp[1][0], pool_type='max')

    # output layer
    feature_out = fluid.layers.fc(input=[fc_last, lstm_last], size=conf_dict['class_dim'])

    return feature_out
