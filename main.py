from random import random

import tensorflow as tf
import numpy as np
import os
import time
import json

from keras import layers
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from django.http import HttpResponse
from flask import Flask, render_template, url_for, request

app = Flask(__name__)
picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder


@app.route('/')
@app.route('/home')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard1.png')
    pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard2.png')
    pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard3.png')
    pic4 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard4.png')
    pic5 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard5.png')
    pic6 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard6.png')
    return render_template("index.html", dashboard1=pic1, dashboard2=pic2, dashboard3=pic3, dashboard4=pic4,
                           dashboard5=pic5, dashboard6=pic6)


@app.route('/result', methods=['POST', 'GET'])
def result():
    output = request.form["graph"]
    second = request.form["company"]
    print(second)
    print(output)
    name = run_prediction(output, second)

    if second == "microsoft":
        if "Balance" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard1.png')
        if "Retained" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard2.png')
        if "Income" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard3.png')
        if "Expenses" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard4.png')
        if "Stock" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard5.png')
        if "Share" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard6.png')
        print(name)

        pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard1.png')
        pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard2.png')
        pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard3.png')
        pic4 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard4.png')
        pic5 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard5.png')
        pic6 = os.path.join(app.config['UPLOAD_FOLDER'], 'Dashboard6.png')
        return render_template('index.html', name=name, dashboard1=pic1, dashboard2=pic2, dashboard3=pic3,
                               dashboard4=pic4, dashboard5=pic5, dashboard6=pic6, type=type)
    if second == "apple":
        if "Balance" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard1.png')
        if "Retained" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard2.png')
        if "Income" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard3.png')
        if "Expenses" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard4.png')
        if "Stock" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard5.png')
        if "Share" in output:
            type = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard6.png')
        print(name)

        pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard1.png')
        pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard2.png')
        pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard3.png')
        pic4 = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard4.png')
        pic5 = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard5.png')
        pic6 = os.path.join(app.config['UPLOAD_FOLDER'], 'ADashboard6.png')
        return render_template('index.html', name=name, dashboard1=pic1, dashboard2=pic2, dashboard3=pic3,
                               dashboard4=pic4, dashboard5=pic5, dashboard6=pic6, type=type)


# text = webtext.raw('overheard.txt')
# text = gutenberg.raw('bible-kjv.txt')
file = open('training_text.txt', encoding="utf-8")

text = file.read()

print(text[:200])

vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, start_string, model_name):

    if model_name == "microsoft":

        file = open('training_text.txt', encoding="utf-8")

        text = file.read()

        print(text[:200])

        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        num_generate = 100

        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        temperature = 1.0

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)

            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        text = start_string + ''.join(text_generated)

        return text

    if model_name == "apple":

        file = open('apple.txt', encoding="utf-8")

        text = file.read()

        print(text[:200])

        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        num_generate = 100

        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        temperature = 1.0

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        text = start_string + ''.join(text_generated)

        return text


def train():
    file = open('apple.txt', encoding="utf-8")

    text = file.read()

    print(text[:200])

    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    seq_length = 100
    examples_per_epoch = len(text) // (seq_length + 1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))


    BATCH_SIZE = 64

    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(vocab)

    embedding_dim = 256

    rnn_units = 2048

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    # Directory where the checkpoints will be saved
    checkpoint_dir = './apple_models'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "best_model")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        monitor='loss',
        mode='min',
        verbose=1,
        save_best_only='True',
        save_weights_only=True)

    model.compile(optimizer='adam', loss=loss)
    history = model.fit(dataset, epochs=15, callbacks=[checkpoint_callback])

    model.save('apple_model.h5', save_format='h5')

    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    model.save('apple_model.h5', save_format='h5')

    text = generate_text(model, start_string=u"2021 Revenue")

    return text


def run_prediction(seed, model_name):
    if model_name == 'microsoft':

        file = open('training_text.txt', encoding="utf-8")

        text = file.read()

        print(text[:200])

        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        year = seed[-5:]
        sheet = seed[:-6]

        labels = []
        year1 = ""
        year2 = ""
        if year == "19-21":
            year1 = "2019"
            year2 = "2021"
        if year == "20-21":
            year1 = "2020"
            year2 = "2021"

        if sheet == 'Balance Sheet':
            with open('GraphDict1.json') as json_file:
                dict = json.load(json_file)
            print(dict["0"]['Label'])
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])
            labels.append(dict["2"]['Label'])
        if sheet == 'Retained Earnings':
            with open('GraphDict2.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict['Label'])
        if sheet == 'Income':
            with open('GraphDict3.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])
            labels.append(dict["2"]['Label'])
        if sheet == 'Expenses':
            with open('GraphDict4.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])
            labels.append(dict["2"]['Label'])
            labels.append(dict["3"]['Label'])
            labels.append(dict["4"]['Label'])
            labels.append(dict["5"]['Label'])
        if sheet == 'Stock Chart':
            with open('GraphDict5.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict['Label'])
        if sheet == 'Earnings Per Share':
            with open('GraphDict6.json') as json_file:
                dict = json.load(json_file)
            print(dict["0"]['Label'])
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])

        print(labels)

        vocab_size = len(vocab)

        embedding_dim = 256

        rnn_units = 2048

        checkpoint_dir = './models'

        checkpoint_prefix = os.path.join(checkpoint_dir, "best_model")

        tf.train.latest_checkpoint(checkpoint_dir)
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        model.build(tf.TensorShape([1, None]))

        # model.save('model.h5', save_format='h5')
    if model_name == 'apple':

        file = open('apple.txt', encoding="utf-8")

        text = file.read()

        print(text[:200])

        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        year = seed[-5:]
        sheet = seed[:-6]

        labels = []
        year1 = ""
        year2 = ""
        if year == "19-21":
            year1 = "2019"
            year2 = "2021"
        if year == "20-21":
            year1 = "2020"
            year2 = "2021"

        if sheet == 'Balance Sheet':
            with open('AGraphDict1.json') as json_file:
                dict = json.load(json_file)
            print(dict["0"]['Label'])
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])
            labels.append(dict["2"]['Label'])
        if sheet == 'Retained Earnings':
            with open('AGraphDict2.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict['Label'])
        if sheet == 'Income':
            with open('AGraphDict3.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])
            labels.append(dict["2"]['Label'])
        if sheet == 'Expenses':
            with open('AGraphDict4.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])
            labels.append(dict["2"]['Label'])
            labels.append(dict["3"]['Label'])
            labels.append(dict["4"]['Label'])
        if sheet == 'Stock Chart':
            with open('AGraphDict5.json') as json_file:
                dict = json.load(json_file)
            labels.append(dict['Label'])
        if sheet == 'Earnings Per Share':
            with open('AGraphDict6.json') as json_file:
                dict = json.load(json_file)
            print(dict["0"]['Label'])
            labels.append(dict["0"]['Label'])
            labels.append(dict["1"]['Label'])

        print(labels)

        vocab_size = len(vocab)

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 2048

        checkpoint_dir = './apple_models'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "best_model")

        tf.train.latest_checkpoint(checkpoint_dir)
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        model.build(tf.TensorShape([1, None]))

    full_text = ""

    for label in labels:
        print(label)
        string = "Over the time period of " + year1 + " to " + year2 + " " + label + ""
        print(string)
        text = generate_text(model, start_string=string, model_name=model_name)
        full_text = full_text + ".\n " + text
    return full_text


def get():
    return ('hello')


if __name__ == '__main__':
    # train()
    # run_prediction()
    app.run(debug=True, port=5002)
