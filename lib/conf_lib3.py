"""
This module to laod the configuration file
"""

import json
import os
import sys
import argparse
import configparser


def load_conf(conf_filename):
    """
    load the conf file
    :param string conf_filename: conf file
    :rtype dict param_conf_dict: conf_dict
    """
    param_conf_dict={}
    cf = configparser.ConfigParser()
    cf.read(conf_filename)
    int_conf_keys = {
                'model_params': ["cost_threshold", "mark_dict_len", "word_dim",
                    "mark_dim", "postag_dim", "hidden_dim", "depth",
                    "pass_num", "batch_size", "class_dim"]
                }
    for session_key in int_conf_keys:
        for option_key in int_conf_keys[session_key]:
            try:
                option_value = cf.get(session_key, option_key)
                param_conf_dict[option_key] = int(option_value)
            except:
                raise ValueError("%s--%s is not a integer" % (session_key, option_key))
    str_conf_keys = {
                'model_params': ['is_sparse', "use_gpu", "emb_name",
                    "is_local", "word_emb_fixed", "mix_hidden_lr"],
                'p_model_dir': ["test_data_path", "train_data_path",
                    "p_model_save_dir"],
                'spo_model_dir': ['spo_test_data_path', 'spo_train_data_path',
                    'spo_model_save_dir'],
                'dict_path': ["so_label_dict_path", "label_dict_path","postag_dict_path", "word_idx_path",
                              "arc_relation_dict_path"]
                }

    for session_key in str_conf_keys:
        for option_key in str_conf_keys[session_key]:
            try:
                param_conf_dict[option_key] = cf.get(session_key, option_key)
            except:
                raise ValueError("%s no such option %s" % (session_key, option_key))
    return param_conf_dict
