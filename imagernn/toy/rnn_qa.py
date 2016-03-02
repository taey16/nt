import numpy as np
import sys
import scipy.io as sio

def sample(hprev, xt, n):
  for index in xt:
    x = np.zeros((vocab_size, 1))
    x[index] = 1
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hprev) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    hprev = h

  generated_seq = []
  generated_seq.append(ix)
  x = np.zeros((vocab_size, 1))
  x[ix] = 1
  h = hprev
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    generated_seq.append(ix)
  return generated_seq

def generate_h(xt):
  hidden_size = Whh.shape[0]
  hprev = np.zeros((hidden_size,1))
  for index in xt:
    x = np.zeros((vocab_size, 1))
    x[index] = 1
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hprev) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    hprev = h
  return hprev

if __name__ == '__main__':

  dataset = 'lstm_from_wiki.txt'
  data = open('data/%s' % dataset, 'r').read()
  chars = list(set(data))
  print '%d unique characters in data.' % (len(chars))
  vocab_size, data_size = len(chars), len(data)
  char_to_ix = { ch:i for i,ch in enumerate(chars) }
  ix_to_char = { i:ch for i,ch in enumerate(chars) }

  # nb. of sequence to generate from model
  seq_length_sample = 2048

  model = sio.loadmat(sys.argv[1])
  p, _, Wxh, Whh, Why, bh, by = \
    model['p'], model['hprev'], model['Wxh'], model['Whh'], model['Why'], model['bh'], model['by']
  inputs = [char_to_ix[ch] for ch in data[1:p]]
  hidden_size = Whh.shape[0]

  while (True):
    question = raw_input("Inital seq.: ")
    print "gen-seq.: "
    inputs = [char_to_ix[ch] for ch in question]
    # initial state
    hprev = np.zeros((hidden_size,1))
    sample_ix = sample(hprev, inputs, seq_length_sample)
    txt = ''.join(ix_to_char[ix] for ix in inputs) + ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % txt
