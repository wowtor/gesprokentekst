#!/usr/bin/env python3

import collections
import logging
import re
import os
import string

import json
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, Normalizer
from tqdm import tqdm

import hashfiles


LOG = logging.getLogger(__name__)


class extract_words():
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __call__(self, texts):
        newtexts = []
        for text in texts:
            newtexts.append([word for word in text if word in self.wordlist])
        return newtexts


class rearrange_samples():
    def __init__(self, n):
        self.n = n

    def __call__(self, items):
        allwords = [ word for item in items for word in item ]

        nitems = len(allwords) // self.n
        itemsize = len(allwords) / nitems
        resampled_items = []
        for i in range(nitems):
            resampled_items.append(allwords[int(i*itemsize):int(i*itemsize+itemsize)])

        LOG.info('rearrange_samples: %d elements in %d items -> %d items' % (len(allwords), len(items), len(resampled_items)))

        return resampled_items


class create_feature_vector():
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, samples):
        vectorizer = TfidfVectorizer(use_idf=False, norm=None, vocabulary=self.vocabulary)
        samples = [' '.join(words) for words in samples]
        return vectorizer.fit_transform(samples).toarray()


def opentext(path, digest):
    with open(path, 'rb') as f:
        bom = f.read(2)
    encoding = 'utf-16-be' if bom == b'\xfe\xff' else 'ascii'
    return hashfiles.open(path, digest, algorithm='sha256', encoding=encoding, mode='t')


def read_session(lines):
    speakers = []
    for line in lines:
        if line.startswith('    item ['):
            words = []
            speakers.append(words)
        mtext = re.match(' *text = "(.*)"', line)
        if mtext is not None:
            s = mtext.group(1)
            s = s.translate({ord(c): None for c in string.punctuation})
            words.extend(word_tokenize(s))

    return speakers


def compile_data():
    speakers = collections.defaultdict(list)
    for digest, filepath in tqdm(list(hashfiles.read_list('files.sha256')), desc='compiling data'):
        speakerids = re.findall('SP[0-9]{3}', os.path.basename(filepath))
        with opentext(filepath, digest) as f:
            for speakerid, texts in zip(speakerids, read_session(f)[:2]):
                speakers[speakerid].append(texts)

    return speakers


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def store_data(path, speakers):
    with open(path, 'w') as f:
        f.write(json.dumps(speakers))


def get_frequent_words(speakers, n):
    freq = collections.defaultdict(int)
    for speaker in speakers.values():
        for text in speaker:
            for word in text:
                freq[word] += 1

    freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)

    return freq[:n]


def filter_texts(speakerdict, wordlist, aantal_woorden):
    filters = [
        extract_words(wordlist),
        rearrange_samples(aantal_woorden),
        create_feature_vector(wordlist),
    ]

    filtered = {}
    for label, texts in speakerdict.items():
        LOG.info('filter in subset {}'.format(label))
        for f in filters:
            texts = f(texts)
        filtered[label] = texts

    return filtered


def to_vector(speakers):
    labels = []
    features = []
    distinct_labels = sorted(speakers.keys())
    for label, texts in speakers.items():
        labels.extend([distinct_labels.index(label) for i in range(len(texts))])
        features.append(texts)

    return np.concatenate(features), labels


def print_overview(speakers):
    for label, texts in speakers.items():
        print('label: {}; {} texts'.format(label, len(texts)))


if __name__ == '__main__':
    speakers_path = 'speakers.json'
    if os.path.exists(speakers_path):
        print('loading', speakers_path)
        speakers = load_data(speakers_path)
    else:
        speakers = compile_data()
        store_data(speakers_path, speakers)

    wordlist = list(zip(*get_frequent_words(speakers, 100)))[0]
    speakers = filter_texts(speakers, wordlist, 20)

    X, y = to_vector(speakers)
    X_train, X_test, y_train, y_test = train_test_split([X, y], train_size=.8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    
    