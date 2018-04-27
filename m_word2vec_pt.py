import os
import random
import pickle
import zipfile
import argparse
import collections
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from tempfile import gettempdir
from sklearn.preprocessing import normalize
from six.moves import urllib


np.set_printoptions(precision=2)
verbose = 0


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # v_embeddings for center words
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # u_embeddings for context words
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Init parameters for u and v if train from scratch
        self.v_embeddings.weight.data.uniform_(-1.0, 1.0)
        self.u_embeddings.weight.data.normal_()

    def forward(self, pos_v, pos_u, neg_u):
        """
        :param pos_v: (batch_size) long tensor variables of input (center) word ids
        :param pos_u: (batch_size) long tensor variables of label (context) word ids
        :param neg_u: (batch_size, num_neg) long tensor variables of negative word ids for each pos_v
        :return: The cross entropy loss between u and v
                 minimize L  = -log[exp(pos_u'*pos_v) / sum_i(exp(neg_u'*pos_v))]
                             =  log(sum_i(exp(neg_u'*pos_v))) - pos_u'*pos_v
                             =  log(neg_score) - pos_score
        """
        pos_v_embedding = self.v_embeddings(pos_v)
        pos_u_embedding = self.u_embeddings(pos_u)
        neg_u_embedding = self.u_embeddings(neg_u)

        # Similarity between pos_u and pos_v, i.e., pos_u'*pos_v (dot product between pos_u and pos_v)
        # pos_u_embedding: (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        # pos_v_embedding: (batch_size, embedding_dim) -> (batch_size, embedding_dim, 1)
        # pos_score: (batch_size) each value is pos_u'*pos_v
        pos_score = torch.bmm(pos_u_embedding.view(-1, 1, self.embedding_dim),
                              pos_v_embedding.view(-1, self.embedding_dim, 1)).squeeze()
        if verbose:
            print 'pos_score:', pos_score.size()
            print pos_score.data

        # Similarity between neg_u and pos_v, i.e., dot product between each neg_u and pos_v
        # neg_u_embedding: (batch_size, num_neg, embedding_dim)
        # pos_v_embedding: (batch_size, embedding_dim) -> (batch_size, embedding_dim, 1)
        # neg_score: (batch_size, num_neg) each value is neg_u'*pos_v
        neg_score = torch.bmm(neg_u_embedding, pos_v_embedding.view(-1, self.embedding_dim, 1)).squeeze()

        if verbose:
            print 'neg_score:', neg_score.size()
            print neg_score.data

        # Compute log(sum_i(exp(neg_u'*pos_v)))
        # neg_score: (batch_size, num_neg)
        # log_neg_score: (batch_size) 
        log_neg_score = torch.log(torch.sum(torch.exp(neg_score), dim=1))
        
        # Compute final score then take average, i.e., score = log(neg_score) - pos_score
        # log_neg_score: (batch_size)
        # pos_score: (batch_size)
        # score: scalar for loss
        score = log_neg_score - pos_score
        score = torch.mean(score)

        return score

    def forward_u(self, u):
        return self.u_embeddings(u)

    def forward_v(self, v):
        return self.v_embeddings(v)

    def set_u_embeddings(self, weights):
        self.u_embeddings.weight.data.copy_(torch.from_numpy(weights))

    def set_v_embeddings(self, weights):
        self.v_embeddings.weight.data.copy_(torch.from_numpy(weights))

    def get_u_embeddings(self):
        return self.u_embeddings.weight.data

    def get_v_embeddings(self):
        return self.v_embeddings.weight.data


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


def read_data(file_name):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(file_name) as f:
        vocabs = (f.read(f.namelist()[0])).split()
    return vocabs


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
    parser.add_argument('--gpuid', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vocabulary_size', type=int, default=100000)
    parser.add_argument('--embedding_size', type=int, default=128)

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

    print 'Step 2: Build words dictionary'
    data, count, dictionary, reverse_dictionary = build_dataset(words, args.vocabulary_size)
    del words  # Hint to reduce memory.
    print '\tMost common words (+UNK)', count[:5]
    print '\tSample data', data[:10], [reverse_dictionary[i] for i in data[:10]]

    index = 0
    print '\tData index:', index

    print 'Step 3: Test function for generating training batch for the skip-gram model'
    batch, labels, _ = generate_batch(data=data, index=index, batch_sz=8, n_skips=2, skip_sz=1)
    for i in range(8):
        print '\t', batch[i], reverse_dictionary[batch[i]], '->', labels[i], reverse_dictionary[labels[i]]

    print 'Step 4: Build Skip Gram model'
    net = SkipGram(args.vocabulary_size, args.embedding_size)
    
    # Load pretrained weights
    pretrained_file = './weights/w2v.model'
    if os.path.exists(pretrained_file):
        v_weights, u_weights = pickle.load(open(pretrained_file, 'r'))
        net.set_v_embeddings(v_weights)
        net.set_u_embeddings(u_weights)
    
    # Optimizer for parameters
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)

    # Cuda
    if args.cuda:
        net.cuda()

    # valid examples for checking
    valid_examples = np.random.choice(args.valid_window, args.valid_size, replace=False)
    batch_valid = torch.LongTensor(valid_examples)
    batch_valid = Variable(batch_valid.cuda()) if args.cuda else Variable(batch_valid)

    print 'Step 5: Train Skip Gram model'
    avg_loss = 0

    for step in xrange(args.num_steps):
        batch_mids, batch_lbls, index = generate_batch(data, index, args.batch_size, args.num_skips, args.skip_window)

        # batch_inps: 1d array: (batch_size)
        batch_inps = np.squeeze(batch_mids)

        # batch_tgts: 1d array: (batch_size)
        batch_tgts = np.squeeze(batch_lbls)

        # batch_negs: 2d array: (batch_size, num_neg) in this case, we use other pos_v as neg_v
        batch_negs = np.repeat(np.expand_dims(batch_lbls, 1).transpose(), batch_lbls.shape[0], axis=0)
        # batch_negs = np.tile(np.repeat(np.expand_dims(batch_lbls, 1).transpose(), batch_lbls.shape[0], axis=0), 2)

        if verbose:
            print 'batch_inps:', batch_inps.shape
            print batch_inps
            print 'batch_tgts:', batch_tgts.shape
            print batch_tgts
            print 'batch_negs:', batch_negs.shape
            print batch_negs

        # To long tensor
        batch_inps = torch.LongTensor(batch_inps)
        batch_tgts = torch.LongTensor(batch_tgts)
        batch_negs = torch.LongTensor(batch_negs)

        # Cuda
        batch_inps = Variable(batch_inps.cuda()) if args.cuda else Variable(batch_inps)
        batch_tgts = Variable(batch_tgts.cuda()) if args.cuda else Variable(batch_tgts)
        batch_negs = Variable(batch_negs.cuda()) if args.cuda else Variable(batch_negs)

        # Zero gradient
        net.zero_grad()

        # Forward and get loss
        loss = net(batch_inps, batch_tgts, batch_negs)

        # Backward
        loss.backward()

        # Step the optimizer
        optimizer.step()

        avg_loss += loss.data[0]
        
        if step % args.avg_step == 0:
            if step > 0:
                avg_loss /= args.avg_step
            # The average loss is an estimate of the loss over the last 2000 batches.
            print '\tAverage loss at iter %6d:' % step, avg_loss
            avg_loss = 0

        if step % args.ckpt_step == 0:
            # Get embeddings of valid words and perform L2-normalization
            valid_embeddings = net.forward_v(batch_valid)
            valid_embeddings = valid_embeddings.data.cpu().numpy() if args.cuda else valid_embeddings.data.numpy()
            valid_embeddings = normalize(valid_embeddings, norm='l2', axis=1)

            # Get embeddings of all words and perform L2-normalization
            embeddings = net.get_v_embeddings().cpu().numpy() if args.cuda else net.get_v_embeddings().numpy()
            normalized_embeddings = normalize(embeddings, norm='l2', axis=1)

            # Compute cosine similarity between valid words and all words in dictionary
            sim = np.matmul(valid_embeddings, np.transpose(normalized_embeddings))

            # Print top-k neighbors for each valid word
            for i in xrange(args.valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]

                log_str = '\tNearest to %-10s :' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %-16s' % (log_str, close_word)
                print log_str

    print 'Step 6: Save the trained model'
    embeddings = net.get_v_embeddings().cpu().numpy() if args.cuda else net.get_v_embeddings().numpy()
    final_embeddings = normalize(embeddings, norm='l2', axis=1)

    model_dir = './models'
    maybe_create_path(model_dir)
    model_path = os.path.join(model_dir, 'word2vec_pt.model')

    print '\tSaving trained weights to %s' % model_path
    pickle.dump([final_embeddings, dictionary, reverse_dictionary], open(model_path, 'w'))

if __name__ == '__main__':
    main()

