import pandas as pd
import os
from utils.data_utils import shuffle, pad_sequences
from abcnn import args


# 加载训练数据
def load_my_data(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['string1'].values[0:data_size]
    h = df['string2'].values[0:data_size]
    label = df['score'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    return p_c_index, h_c_index, label


def make_vocab(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['string1'].values[0:data_size]
    h = df['string2'].values[0:data_size]

    vocab = set()
    for item in p:
        words = item.strip().split()
        for w in words:
            vocab.add(w)
    for item in h:
        words = item.strip().split()
        for w in words:
            vocab.add(w.strip())


# 加载字典
def load_char_vocab():
    path = os.path.join(os.path.dirname(__file__), '../input/vocab.txt')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


# 加载词典
def load_word_vocab():
    path = os.path.join(os.path.dirname(__file__), '../output/word2vec/word_vocab.tsv')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


# 字->index
def char_index(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.seq_length)
    h_list = pad_sequences(h_list, maxlen=args.seq_length)

    return p_list, h_list


# 词->index
def word_index(p_sentences, h_sentences):
    word2idx, idx2word = load_word_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.seq_length)
    h_list = pad_sequences(h_list, maxlen=args.seq_length)

    return p_list, h_list


# 加载char_index训练数据
def load_char_data(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    return p_c_index, h_c_index, label


if __name__ == '__main__':

    pass
