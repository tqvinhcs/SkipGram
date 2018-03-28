import os
import math
import random
import pickle
import zipfile
import argparse
import collections

import numpy as np
import tensorflow as tf

from tempfile import gettempdir
from six.moves import urllib


class SkipGram():
    def __init__(self, loss_model, batch_size, vocabulary_size, embedding_size, num_sampled, valid_examples):
        # Dimension information
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Define variables for Word2Vec model
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        self.valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Look up embeddings for inputs.
        self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        stddev = 1.0 / math.sqrt(embedding_size)
        self.u_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=stddev))

        # Get context embeddings from labels
        labels = tf.nn.embedding_lookup(self.u_weights, self.train_labels)
        labels = tf.reshape(labels, [-1, embedding_size])

        if loss_model == 'cross_entropy':
            # Calculate local cross entropy loss (local means using other inputs as negative samples)
            self.loss = tf.reduce_mean(SkipGram.cross_entropy(inputs, labels))
        else:
            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=stddev))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                      biases=nce_biases,
                                                      labels=self.train_labels,
                                                      inputs=inputs,
                                                      num_sampled=num_sampled,
                                                      num_classes=vocabulary_size))

        # Compute the cosine similarity between mini-batch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = tf.div(self.embeddings, norm)

        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

    def set_embeddings(self, weights, sess):
        self.embeddings.load(weights, session=sess)

    def set_u_weights(self, weights, sess):
        self.u_weights.load(weights, session=sess)

    @staticmethod
    def cross_entropy(inputs, labels):
        """ Compute cross entropy loss between inputs and targets
        :param inputs: The embeddings for center words. Dimension is (batch_size, embedding_size).
        :param labels: The embeddings for context words. Dimension is (batch_size, embedding_size).
        :return:  A scalar representing the mean of cross entropy loss
        """

        # Pairwise dot product between inputs ans targets
        prod = tf.matmul(inputs, labels, transpose_b=True)

        # Compute numerator = log(exp({u_o}^T v_c)) by getting the diagonal of prod then take average over batch
        numerator = tf.reduce_mean(tf.diag_part(prod))

        # Compute denominator = log(\sum{exp({u_w}^T v_c)}), by sum each exp(row), take log, then average over batch
        denominator = tf.reduce_mean(tf.log(tf.reduce_sum(tf.exp(prod), 1)))

        return tf.subtract(denominator, numerator)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def maybe_create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print "Created a path: %s" % path


def maybe_download(file_name, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    url = 'http://mattmahoney.net/dc/'
    local_file_name = os.path.join(gettempdir(), file_name)

    if not os.path.exists(local_file_name):
        local_file_name, _ = urllib.request.urlretrieve(url + file_name, local_file_name)

    stat_info = os.stat(local_file_name)

    if stat_info.st_size == expected_bytes:
        print 'Found and verified', file_name
    else:
        print stat_info.st_size
        raise Exception('Failed to verify' + local_file_name + '. Can you get to it with a browser?')

    return local_file_name


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        words = (f.read(f.namelist()[0])).split()
    return words


def build_dataset(words, num_words):
    """
    Building the dataset for Skip Gram model
    :param words: 1d Array of words
    :param num_words: The number of words in the dictionary
    :return:
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(num_words - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary


def generate_batch(data, index, batch_sz, n_skips, skip_sz):
    assert batch_sz % n_skips == 0
    assert n_skips <= 2 * skip_sz

    inputs = np.ndarray(shape=batch_sz, dtype=np.int64)
    labels = np.ndarray(shape=batch_sz, dtype=np.int64)

    span = 2 * skip_sz + 1
    buff = collections.deque(maxlen=span)

    if index + span > len(data):
        index = 0

    buff.extend(data[index:index + span])
    index += span

    for i in range(batch_sz // n_skips):
        context_words = [w for w in range(span) if w != skip_sz]
        words_to_use = random.sample(context_words, n_skips)

        for j, context_word in enumerate(words_to_use):
            inputs[i * n_skips + j] = buff[skip_sz]
            labels[i * n_skips + j] = buff[context_word]

        if index == len(data):
            buff.extend(data[0:span])
            index = span
        else:
            buff.append(data[index])
            index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    index = (index + len(data) - span) % len(data)
    return inputs, labels, index


def main():
    # Step 0: Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str2bool, default='true')
    parser.add_argument('--gpuid', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--vocabulary_size', type=int, default=100000)
    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--loss_model', type=str, default='cross_entropy')
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=100001)
    parser.add_argument('--avg_step', type=int, default=2000)
    parser.add_argument('--ckpt_step', type=int, default=10000)

    parser.add_argument('--skip_window', type=int, default=1)
    parser.add_argument('--num_skips', type=int, default=2)
    parser.add_argument('--num_sampled', type=int, default=64)

    parser.add_argument('--valid_size', type=int, default=16)
    parser.add_argument('--valid_window', type=int, default=100)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

    print 'Step 1: Download the data'
    file_name = maybe_download('text8.zip', 31344016)

    words = read_data(file_name)
    print '\tData size:', len(words)

    print 'Step 2: Build the dictionary'
    data, count, dictionary, reverse_dictionary = build_dataset(words, args.vocabulary_size)
    del words  # Hint to reduce memory.
    print '\tMost common words (+UNK)', count[:5]
    print '\tSample data:', data[:10], [reverse_dictionary[i] for i in data[:10]]

    data_index = 0
    print '\tData index:', data_index

    print 'Step 3: Test function for generating training batch for the skip-gram model'
    batch, labels, _ = generate_batch(data=data, index=data_index, batch_sz=8, n_skips=2, skip_sz=1)
    for i in range(8):
        print '\t', batch[i], reverse_dictionary[batch[i]], '->', labels[i], reverse_dictionary[labels[i]]

    valid_examples = np.random.choice(args.valid_window, args.valid_size, replace=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print 'Step 4: Build Skip Gram model'
        model = SkipGram(args.loss_model, args.batch_size, args.vocabulary_size, args.embedding_size, args.num_sampled, valid_examples)

        # Construct the optimizer.
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss=model.loss, global_step=global_step)

        # Init all global variables
        tf.global_variables_initializer().run()

        # Load pretrained weights
        pretrained_file = './weights/w2v.model'
        if os.path.exists(pretrained_file):
            embeddings, u_weights = pickle.load(open(pretrained_file, 'r'))
            model.set_embeddings(weights=embeddings, sess=sess)
            model.set_u_weights(weights=u_weights, sess=sess)

        average_loss = 0
        print 'Step 5: Train Skip Gram model'
        for step in xrange(args.num_steps):
            batch_inps, batch_tgts, data_index = generate_batch(data, data_index, args.batch_size, args.num_skips, args.skip_window)
            batch_inps = np.squeeze(batch_inps)
            batch_tgts = np.expand_dims(np.squeeze(batch_tgts), axis=1)

            feed_dict = {model.train_inputs: batch_inps, model.train_labels: batch_tgts}

            _, loss_val = sess.run([optimizer, model.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % args.avg_step == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print '\tAverage loss at iter', step, ':', average_loss
                average_loss = 0

            # Print top-k neighbors for each valid word
            if step % args.ckpt_step == 0:
                sim = model.similarity.eval()
                for i in xrange(args.valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]

                    log_str = '\tNearest to %-10s :' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %-16s" % (log_str, close_word)

                    print log_str

        print 'Step 6: Save the trained model'
        final_embeddings = model.normalized_embeddings.eval()

        model_dir = './models'
        maybe_create_path(model_dir)
        model_path = os.path.join(model_dir, 'word2vec_tf.model')

        print '\tSaving trained model to %s' % model_path
        pickle.dump([final_embeddings, dictionary, reverse_dictionary], open(model_path, 'w'))


if __name__ == '__main__':
    main()
