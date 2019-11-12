#!/usr/bin/env python
import os
from itertools import islice, chain
import pickle

import keras
from keras import backend as K
import tensorflow as tf
from tensorflow import math as tfm
import numpy as np
import argparse
import sys
from InputParsers import tempeval2
#from InputParsers import false_data as tempeval2
from time import perf_counter
import re
from copy import deepcopy
from pretrained_embed import encode_word
from pretrained_embed import pretrained_embedding_layer
#from pretrained_embed import embedding_layer, false_data
#pretrained_embedding_layer = lambda x: embedding_layer(x, false_data)

from collections import deque

from hashlib import md5

from multiprocessing import Pool

import json

from kfold import ikfold, average_metrics
from itertools import tee

trial_data_path = './data/tempeval2-trial/data/english'
train_data_path = './data/tempeval-training-2/english/data'

def savemodel(model, filename):
    print("Saving to "+filename)
    model.save(filename, overwrite=True)

def loadmodel(filename):
    print(filename)
    model = keras.models.load_model(filename)
    return model

def encode(words, word_index):
    return [encode_word(word, word_index) for word in words]

def to_windows(s, window_size, word_index):
    tokens = s.split(' ')
    windows = []
    for t in range(len(tokens)):
        frame = ['']*window_size
        limit = (window_size-1)>>1
        for i in range(window_size):
            delta = i-limit
            try:
                index = t + delta
                if index<0:
                    raise IndexError()
                frame[i] = tokens[t+delta]
            except IndexError:
                pass
        windows.append(encode(frame, word_index))
    #print(np.array(windows))
    return np.array(windows)

def load_training_data(window_size, path, word_index, data_generator):
    for s in data_generator:
        windows = []
        event_labels = []
        timex_labels = []
        for t in range(len(s)):
            frame = ['']*window_size
            event_label_frame = [0]*window_size
            timex_label_frame = [0]*window_size
            limit = (window_size-1)>>1
            for i in range(window_size):
                delta = i-limit
                try:
                    index = t + delta
                    if index<0:
                        raise IndexError()
                    frame[i] = s[t+delta]['token']
                    if 'events' in s[t+delta]:
                        event_label_frame[i] = 1
                    if 'timexes' in s[t+delta]:
                        timex_label_frame[i] = 1
                except IndexError:
                    pass
            windows.append(encode(frame, word_index))
            event_labels.append(event_label_frame)
            timex_labels.append(timex_label_frame)
        item = (np.array(windows), [(np.array(event_labels)), (np.array(timex_labels))], ' '.join([t['token']for t in s]))
        yield item

def training_data(window_size, path, word_index, data_generator, get_text=False):
    data = load_training_data(window_size, path, word_index, data_generator)
    if get_text:
        for item in data:
            yield item
    else:
        for item in data:
            yield item[0:2]

def training_data_flat(window_size, path, word_index, data_generator, get_text=False, infinite=False):
    middle = (window_size - 1)//2
    while True:
        data_generator, next_gen = tee(data_generator)
        for item in training_data(window_size, path, word_index, data_generator, True):
            data, (events, timexes), text = item
            inputs = []
            oevents = []
            otimexes = []
            for i, (datum, event, timex) in enumerate(zip(data, events, timexes)):
                i -= (len(datum)-1)//2
                if i < 0:
                    subtext = text.split(' ')[0:len(datum)+i]
                else:
                    subtext = text.split(' ')[i:i+len(datum)]
                #result = (np.array([datum]), [np.array([event]), np.array([timex])])
                inputs.append(datum)
                otimexes.append([timex[middle]])
                #print(subtext)
                #print(oevents[-1])
                #print(otimexes[-1])
            #batch = (np.array(inputs), [np.array(oevents), np.array(otimexes)])
            batch = (np.array(inputs), np.array(otimexes))
            yield batch
        data_generator = next_gen
        if not infinite:
            break

def training_data_really_flat(window_size, path, word_index, data_generator, get_text=False, infinite=False, batches=False):
    for in_batch, out_batch_t in training_data_flat(window_size, path, word_index, data_generator, get_text=False, infinite=infinite):
        for _in, _out_t in zip(in_batch, out_batch_t):
            if batches:
                b = (np.array([_in]), np.array([_out_t]))
            else:
                b = (_in, _out_t)
            yield b

def get_prediction(model, data):
    p_timexes = model.predict(data)
    return p_timexes

def _div(a,b):
    try:
        return a/b
    except ZeroDivisionError:
        return float('inf')

def other_metrics(model, window_size, word_index, data_generator):
    tp = 0
    fp = 0
    fn = 0
    tot = 0
    for data, timexes in training_data_flat(window_size, '', word_index, data_generator, infinite=False):
        p_timexes = np.round(get_prediction(model, data))
        for key, resp in zip(timexes.flatten(), p_timexes.flatten()):
            tot += 1
            resp = round(resp)
            if key == 1 and resp == 1:
                tp += 1
            if key == 0 and resp == 1:
                fp += 1
            if key == 1 and resp == 0:
                fn += 1

    precision = _div(tp, (tp + fp))
    recall = _div(tp, (tp + fn))
    f_measure = 2 * _div((precision * recall), (precision + recall))

    return {
        'p': precision,
        'r': recall,
        'f': f_measure,
    }

Model = keras.models.Model
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
SimpleRNN = keras.layers.SimpleRNN
LSTM = keras.layers.LSTM
Embedding = keras.layers.Embedding
Input = keras.layers.Input
TimeDistributed = keras.layers.TimeDistributed
Flatten = keras.layers.Flatten

def construct_model(window_size):
    # define the model
    text_input = Input(shape=(window_size,), name='text_in')
    embedding, word_index = pretrained_embedding_layer(window_size)
    embedding = embedding(text_input)
    dense1 = Dense(3, activation='sigmoid')(Flatten()(Dense(5, activation='sigmoid')(embedding)))

    output_timex = Dense(1, activation='sigmoid',  name='timex_output')(dense1)

    # compile the model
    model = Model(inputs=[text_input], outputs=[output_timex])
    model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    return model, word_index


def learn_model(data_generator, test_generator, window_size=7, output_folder='model', lossfunc='categorical_crossentropy', quieter=False):
    assert window_size & 1, "Window size must be odd"
    file_list = sorted(list(os.listdir(output_folder)))
    model_files = [os.path.join(output_folder, f) for f in file_list]
    print('window size', window_size)
    if len(model_files) == 0:
        model, word_index = construct_model(window_size)
        restart_from = None
    else:
        model = loadmodel(model_files[-1])
        _, word_index = pretrained_embedding_layer(window_size)
        num_re = re.compile(r"[^0-9]*([0-9]+)\.h5$")
        restart_from = int(num_re.match(file_list[-1]).groups(1)[0])
    plot_model = keras.utils.plot_model
    plot_model(model, to_file='time_extract.png')

    # summarize the model
    print(model.summary())

    curvex = []
    curvey = []
    curvey_eval = []
    x = 0
    done = False
    if restart_from is not None:
        pos = restart_from
        print("Restart training from epoch %d" % (pos))
    else:
        pos = -1

    data_generator, this_gen = tee(data_generator)
    n_items = len(list(training_data_really_flat(window_size, train_data_path, word_index, this_gen, get_text=False, infinite=False)))
    print(n_items)
    done_epochs=0
    history = []
    while not done:
        start = perf_counter()
        pos += 1
        data_generator, this_gen = tee(data_generator)
        if quieter:
            verb = 0
        else:
            verb = 1
        model.fit_generator(training_data_really_flat(window_size, train_data_path, word_index, this_gen, get_text=False, infinite=True, batches=True), steps_per_epoch=n_items, epochs=1, verbose=verb)
        if pos % 1 == 0:
            print(model.metrics_names)
            data_generator, this_gen = tee(data_generator)
            metrics_train = model.evaluate_generator(training_data_flat(window_size, train_data_path, word_index, this_gen), steps=100)
            test_generator, this_gen = tee(test_generator)
            metrics_eval = model.evaluate_generator(training_data_flat(window_size, trial_data_path, word_index, this_gen), steps=100)
            print(metrics_train)
            print(metrics_eval)
            test_generator, this_gen = tee(test_generator)
            others = other_metrics(model, window_size, word_index, this_gen)
            print("~~TEST~~~~~~~~~~~~~~~~~~~")
            print("precision     ", others['p'])
            print("recall        ", others['r'])
            print("f1            ", others['f'])
            testf1 = others['f']

            data_generator, this_gen = tee(data_generator)
            others = other_metrics(model, window_size, word_index, this_gen)
            print("~~TRAIN~~~~~~~~~~~~~~~~~~")
            print("precision     ", others['p'])
            print("recall        ", others['r'])
            print("f1            ", others['f'])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~")
            trainf1 = others['f']

            filename = '%s/time_extract_model_%08d.h5' % (output_folder, pos)
            history.append((trainf1,testf1,filename))

            x += 1
            if all([
                    (others['p']) != float('inf'),
                    (others['r']) != float('inf'),
                    (others['p']) >= 0.95,
                    (others['r']) >= 0.95
                    ]):
                done = True
            savemodel(model,filename)
        end = perf_counter()
        print("Epoch time: %f" % (end-start))
        done_epochs += 1
        if done_epochs >= 20: # Hard limit at 20 epochs
            done=True
        if len(history) > 2:
            this_train, this_test, _ = history[-1]
            last_train, last_test, filename = history[-2]
            last2_train, last2_test, filename2 = history[-3]
            if this_train < last_train and this_test < last_test:
                # Both stopped improving
                done = True
                model = loadmodel(filename)
                print("Test and training have both stopped improving, selected last good model")
            if this_train > last_train and last_train > last2_train and this_test < last_test and last_test < last2_test:
                done = True
                model = loadmodel(filename2)
                print("Starting to overfit training, test score is worsening, selected last good model")
    print("Train, Test")
    for t1, t2, _ in history:
        print(t1, t2)
    savemodel(model,'web_model.h5')
    return model, word_index

def test_model(model, word_index, window_size,  data_generator):
    others = other_metrics(model, window_size, word_index, data_generator)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("precision     ", others['p'])
    print("recall        ", others['r'])
    print("f1            ", others['f'])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    return others

def eval_model(input_string, window_size, vocab_size, model_file):
    model = loadmodel(model_file)
    _, word_index = pretrained_embedding_layer(window_size)
    input_data = to_windows(input_string, window_size, word_index)
    print(input_data)
    p_timexes = get_prediction(model, input_data)
    print(p_timexes)
    print(input_string)
    for outputs, name in [(p_timexes, 'timexes')]:
        print('==============================')
        print(name)
        tokens = input_string.split(' ')
        for t, m in zip(tokens, outputs):
            tl = len(t)
            m = m[0]
            pad_t = "%s%s |%s%s|%s" % (
                t,
                ' ' * (20-tl),
                ("#" * int(round(m * 30))),
                (" " * (30-int(round(m * 30)))),
                " ~!~" if m > 0.5 else ""
            )
            print(pad_t)

def run_experiment(k):
    cross = []
    for i, (train, test) in enumerate(ikfold(tempeval2.sentences(tempeval2.parse(train_data_path)), k)):
        path = 'kfold_test_run_%d' % (i)
        os.system('rm -rf ' + path)
        os.makedirs(path, exist_ok=True)
        test, test2 = tee(test)
        model, word_index = learn_model(train, test, 5, output_folder=path, quieter=True)
        cross.append(test_model(model, word_index, 5, test2))

    return [{'name':'Deep NN'
        ,'cross':cross
        ,**average_metrics(cross)
    }]

if __name__ == '__main__':
    print("start")

    parser = argparse.ArgumentParser("Time Extractor")
    parser.add_argument("--eval")
    parser.add_argument("--modelfile")
    parser.add_argument("--learn", action='store_true')
    parser.add_argument("--test")
    args = parser.parse_args()

    folder = 'models'
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    window_size = 5

    if args.learn:
        print("learn")
        learn_model(tempeval2.sentences(tempeval2.parse(train_data_path)), tempeval2.sentences(tempeval2.parse(trial_data_path)), window_size, output_folder=folder)

    if args.test:
        test_model('TODO', window_size, model_file=args.test)

    if args.eval:
        if not args.modelfile:
            print("Please specify a model file with --modelfile", file=sys.stderr)
            sys.exit(1)
        eval_model(args.eval, window_size, vocab_size, args.modelfile)

