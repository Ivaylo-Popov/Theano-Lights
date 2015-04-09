from collections import Counter
import ConfigParser
import argparse
import os
import time
import sys
import numpy

def construct_vocabulary(dataset, oov_rate, level):
    filename = dataset + '_train'
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', '<eos>')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
    
    # Order the words
    all_items = Counter(txt).items()
    no_end = [x for x in all_items if x[0] !='\n']
    freqs = [x for x in all_items if x[0] == '\n'] + sorted(no_end, key=lambda t: t[1], reverse=True)
    
    # Decide length
    all_freq = float(sum([x[1] for x in freqs]))
    up_to = len(freqs)
    oov = 0.
    remove_word = True
    while remove_word:
        up_to -= 1
        oov += float(freqs[up_to][1])
        if oov / all_freq > oov_rate:
            remove_word = False
    up_to += 1
    freqs = freqs[:up_to]
    words = [x[0] for x in freqs]
    return dict(zip(words, range(up_to))), [x[1]/all_freq for x in freqs], freqs

def grab_text(path, filename, vocab, oov_default, dtype, level):
    filename = path + filename
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' \n ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
        return numpy.asarray(
            [vocab.get(w, oov_default) for w in txt], dtype=dtype)
    else:
        return numpy.array(
            [vocab.get(w, oov_default) for w in txt], dtype=dtype)

def main(parser):
    o = parser.parse_args()
    vocab, freqs, freq_wd = construct_vocabulary(o.path, o.oov_rate, o.level)
    if not '<unk>' in vocab:
        vocab['<unk>'] = numpy.max(list(vocab.values())) + 1
    
    oov_default = vocab["<unk>"]
    train = grab_text(o.path, '_train', vocab, oov_default, o.dtype, o.level)
    valid = grab_text(o.path, '_valid', vocab, oov_default, o.dtype, o.level)
    test = grab_text(o.path, '_test', vocab, oov_default, o.dtype, o.level)

    if o.level == 'words':
        data = {'train_words': train, 'valid_words': valid, 'test_words': test, 'n_words': len(vocab)}
    else:
        data = {'train_chars': train, 'valid_chars': valid, 'test_chars': test, 'n_chars': len(vocab)}

    keys = {'oov': oov_default, 'freqs': numpy.array(freqs), 'vocabulary': vocab, 'freq_wd': freq_wd}
    all_keys = dict(keys.items() + data.items())
    
    numpy.savez(o.path, **all_keys)
    inv_map = [None] * (len(vocab.items()) + 1)
    for k, v in vocab.items():
        inv_map[v] = k

    if o.level == 'words':
        numpy.savez(o.path + "_dict", unique_words=inv_map)
    else:
        numpy.savez(o.path + "_dict", unique_chars=inv_map)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default="ntst")
    parser.add_argument('--level', default='words')
    parser.add_argument('--oov-rate', type=float, default=0.)
    parser.add_argument('--dtype', default='int32')
    return parser

if __name__ == '__main__':
    main(get_parser())
