"""
This module to generate vocabulary list
"""
import os
import json
from tqdm import tqdm

def load_word_file(f_input):
    """
    Get all words in files
    :param string: input file
    """
    file_words = {}
    print("载入文件：")
    with open(f_input, mode='r', encoding='utf-8') as fr:
        data_list = json.load(fr)
        for dic in tqdm(data_list):
            try:
                postag = dic['postag']
                words = [item["word"].strip() for item in postag]
            except:
                continue
            for word in words:
                file_words[word] = file_words.get(word, 0) + 1
    return file_words


def get_vocab(train_file, dev_file):
    """
    Get vocabulary file from the field 'postag' of files
    :param string: input train data file
    :param string: input dev data file
    """
    word_dic = load_word_file(train_file)
    if len(word_dic) == 0:
        raise ValueError('The length of train word is 0')
    dev_word_dic = load_word_file(dev_file)
    if len(dev_word_dic) == 0:
        raise ValueError('The length of dev word is 0')
    for word in dev_word_dic:
        if word in word_dic:
            word_dic[word] += dev_word_dic[word]
        else:
            word_dic[word] = dev_word_dic[word]
    with open("../dict/word_idx", mode='w', encoding='utf-8') as fr_word_idx:
        fr_word_idx.write('<UNK>' + "\n")
    vocab_set = set()
    value_list = sorted(word_dic.items(), key=lambda d: d[1], reverse=True)
    for word in value_list[:30000]:
        with open("../dict/word_idx", mode='a', encoding='utf-8') as fr_word_idx:
            fr_word_idx.write(word[0] + "\n")
        vocab_set.add(word[0])

    # add predicate in all_50_schemas
    if not os.path.exists('../data/all_50_schemas'):
        raise ValueError("../data/all_50_schemas not found.")
    with open('../data/all_50_schemas', mode='r', encoding='utf-8') as fr:
        for line in fr.readlines():
            dic = json.loads(line.strip(), encoding='utf-8')
            p = dic['predicate']
            if p not in vocab_set:
                vocab_set.add(p)
                with open("../dict/word_idx", mode='a', encoding='utf-8') as fr_word_idx:
                    fr_word_idx.write(p + "\n")


if __name__ == '__main__':
    train_file = "../data/train_data_parse.json"
    dev_file = "../data/dev_data_parse.json"
    get_vocab(train_file, dev_file)
