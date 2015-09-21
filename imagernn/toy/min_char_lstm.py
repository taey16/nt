"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import cPickle
import scipy.io as sio
#from IPython.display import clear_output, Image, display
#%matplotlib inline

def sigm(x): 
  return 1. / (1 + np.exp(-x))

def lossFun(inputs, targets, hprev, cprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  loss, pplx = 0, 0
  x_s, y_s, p_s = {}, {}, {}
  g_s, i_s, f_s, o_s, c_s, ct_s, h_s = {}, {}, {}, {}, {}, {}, {} 
  h_s[-1] = np.copy(hprev)
  c_s[-1] = np.copy(cprev)
  # forward pass
  for t in range(len(inputs)):
    # encode in 1-of-k representation
    x_s[t] = np.zeros((vocab_size,1))
    x_s[t][inputs[t]] = 1

    # LSTM
    i_s[t]= sigm(   np.dot(Wxi, x_s[t]) + np.dot(Whi, h_s[t-1]) + bi)
    f_s[t]= sigm(   np.dot(Wxf, x_s[t]) + np.dot(Whf, h_s[t-1]) + bf)
    o_s[t]= sigm(   np.dot(Wxo, x_s[t]) + np.dot(Who, h_s[t-1]) + bo)
    g_s[t]= np.tanh(np.dot(Wxg, x_s[t]) + np.dot(Whg, h_s[t-1]) + bg)
    c_s[t]= f_s[t] * c_s[t-1] + i_s[t] * g_s[t]
    ct_s[t]=np.tanh(c_s[t])
    h_s[t]= o_s[t] * ct_s

    y_s[t] = np.dot(Why, h_s[t]) + by
    # softmax
    p_s[t] = np.exp(y_s[t]) / np.sum(np.exp(y_s[t]))
    # cross-entropy loss
    loss += -np.log(p_s[t][targets[t],0])
    # perplexity
    pplx += -np.log2(p_s[t][targets[t],0])
  pplx = 2 ** (pplx / len(inputs))

  # backward pass: compute gradients going backwards
  # memory variables for derivatives
  dWxg, dWxi, dWxf, dWxo, dWhi, dWhf, dWho, dWhy = np.zeros_like(Wxg), np.zeros_like(Wxi), np.zeros_like(Wxf), np.zeros_like(Wxo), np.zeros_like(Whi), np.zeros_like(Whf), np.zeros_like(Who), np.zeros_like(Why)
  dbg, dbi, dbf, dbo, dby = np.zeros_like(bg), np.zeros_like(bi), np.zeros_like(bf), np.zeros_like(bo), np.zeros_like(by)
  dhnext = np.zeros_like(h_s[0])
  dcnext = np.zeros_like(c_s[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(p_s[t])
    # backprop. into y
    # dLoss = y - t
    dy[targets[t]] -= 1
    # compute grad. w.r.t. Why
    dWhy += np.dot(dy, h_s[t].T)
    # grad. w.r.t. by
    dby += dy

    tanhCt = ct_s[t]

    dh = np.dot(Why.T, dy)

    do = dh * tanhCt
    dc = dh * (1.0 - tanhCt**2)
    di = dc * g_s[t]
    dg = dc * i_s[t]
    df = dc * dcnext

    di_input = (1.0 - i_s[t]) * i_s[t] * di
    df_input = (1.0 - f_s[t]) * f_s[t] * df
    do_input = (1.0 - o_s[t]) * o_s[t] * do
    dg_input = (1.0 - g_s[t]**2) * dg
   
    dWxg += np.dot(dg_input, x_s[t].T)
    dWhg += np.dot(dg_input, h_s[t-1].T)
    dWxi += np.dot(di_input, x_s[t].T)
    dWhi += np.dot(di_input, h_s[t-1].T)
    dWxf += np.dot(df_input, x_s[t].T)
    dWhf += np.dot(df_input, h_s[t-1].T)
    dWxo += np.dot(do_input, x_s[t].T)
    dWho += np.dot(do_input, h_s[t-1].T)

    dbo += do_input
    dbf += df_input
    dbi += di_input
    dbg += dg_input

    dcnext = dc * f_s[t]
    dhnext += np.dot(Whi.T, di_input)
    dhnext += np.dot(Whf.T, df_input)
    dhnext += np.dot(Who.T, do_input)
    dhnext += np.dot(Whg.T, dg_input)

  # clip to mitigate exploding gradients
  for dparam in [dWxg, dWxi, dWxf, dWxo, dWhi, dWhf, dWho, dWhy, dbg, dbi, dbf, dbo, dby]
    dparam = np.clip(dparam, -1, 1, out=dparam)

  return loss, pplx, dWxg, dWxi, dWxf, dWxo, dWhi, dWhf, dWho, dWhy, dbg, dbi, dbf, dbo, dby, h_s[len(inputs)-1], c_s[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  generated_seq = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    generated_seq.append(ix)
  return generated_seq

def rev_sample(h_rec, seed_ix, n):
  p = np.zeros((vocab_size, 1))
  p[seed_ix] = 1
  generated_seq = []
  for t in range(n):
    h_rec = np.dot(Why.T, p) + h_rec
    h_rec = (1 - h_rec**2) * h_rec
    x_rec = np.dot(Wxh.T, h_rec)
    x_rec = (x_rec - np.min(x_rec)) / (np.max(x_rec) - np.min(x_rec))
    x_rec = np.exp(x_rec) / np.sum(np.exp(x_rec))
    ix = np.random.choice(range(vocab_size), p=x_rec.ravel())
    p = x_rec
    h_rec = np.dot(Whh.T, h_rec)
    generated_seq.append(ix)
  return generated_seq

# gradient checking
from random import uniform
def gradCheck(inputs, target, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
    print name
    for i in xrange(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
      # rel_error should be on order of 1e-7 or less

if __name__ == '__main__':

  import pdb; pdb.set_trace()
  # data should be simple plain text file
  data = open('data/lstm_from_wiki.txt', 'r').read()
  #data = open('data/min_char_rnn.data.py', 'r').read()
  chars = list(set(data))
  print '%d unique characters in data.' % (len(chars))
  vocab_size, data_size = len(chars), len(data)
  # compute vocab.
  char_to_ix = { ch:i for i,ch in enumerate(chars) }
  ix_to_char = { i:ch for i,ch in enumerate(chars) }

  # hyperparameters
  # size of hidden layer of neurons
  hidden_size = 256
  # number of steps to unroll the RNN for
  seq_length = 96
  # nb. of sequence to generate from model
  seq_length_sample = 128
  # learning params.
  learning_rate = 0.01
  decay_rate = 0.5

  # model parameters
  Wxg = np.random.randn(hidden_size,vocab_size).astype( np.float32)*0.01
  Wxi = np.random.randn(hidden_size,vocab_size).astype( np.float32)*0.01
  Wxf = np.random.randn(hidden_size,vocab_size).astype( np.float32)*0.01
  Wxo = np.random.randn(hidden_size,vocab_size).astype( np.float32)*0.01
  Whi = np.random.randn(hidden_size,hidden_size).astype(np.float32)*0.01
  Whf = np.random.randn(hidden_size,hidden_size).astype(np.float32)*0.01
  Who = np.random.randn(hidden_size,hidden_size).astype(np.float32)*0.01
  Why = np.random.randn(vocab_size, hidden_size).astype(np.float32)*0.01
  bg = np.zeros((hidden_size,1)).astype(np.float32)
  bi = np.zeros((hidden_size,1)).astype(np.float32)
  bf = np.zeros((hidden_size,1)).astype(np.float32)
  bo = np.zeros((hidden_size,1)).astype(np.float32)
  by = np.zeros((vocab_size,1)).astype(np.float32)
  
  # memory variables for Adagrad
  mWxg, mWxi, mWxf, mWxo, mWhi, mWhf, mWho, mWhy = np.zeros_like(Wxg), np.zeros_like(Wxi), np.zeros_like(Wxf), np.zeros_like(Wxo), np.zeros_like(Whi), np.zeros_like(Whf), np.zeros_like(Who), np.zeros_like(Why)
  mbg, mbi, mbf, mbo, mby = np.zeros_like(bg), np.zeros_like(bi), np.zeros_like(bf), np.zeros_like(bo), np.zeros_like(by)

  n, p, cum_p = 0, 0, 0
  sample_freq = 100

  # compute loss (loss at iter. 0)
  smooth_loss = -np.log(1.0/vocab_size)*seq_length

  # start to train
  while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0: 
      # reset RNN memory
      hprev = np.zeros((hidden_size,1))
      sprev = np.zeros((hidden_size,1))
      p = 0 # go from start of data

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets= [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    """
    # sample from the model now and then
    if n % sample_freq == 0:
      sample_ix = sample(hprev, inputs[0], seq_length_sample)
      txt = ix_to_char[inputs[0]] + ''.join(ix_to_char[ix] for ix in sample_ix)
      print '----\n %s \n----' % txt
      #gradCheck(inputs, targets, hprev)
      #rev_sample_ix = rev_sample(hprev, inputs[0], seq_length_sample)
      #rev_txt = ''.join(ix_to_char[ix] for ix in rev_sample_ix)
      #print '----\n %s \n----' % rev_txt
    """
    # forward seq_length characters through the net and fetch gradient
    loss, pplx, dWxg, dWxi, dWxf, dWxo, dWhi, dWhf, dWho, dWhy, dbg, dbi, dbf, dbo, dby, h_s[len(inputs)-1], c_s[len(inputs)-1] = lossFun(inputs, targets, hprev, sprev)
  
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
    # perform parameter update
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
      ## adagrad
      mem += dparam*dparam
      ## rmsprop
      #mem = (decay_rate * mem) + (1.0 - decay_rate) * (dparam*dparam)
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    # print learning
    if n % sample_freq == 0:
      #clear_output(wait=True)
      #L[n/100] = smooth_loss; plt.plot(L[:n/100+1]); plt.grid(True)
      #display(plt.gcf())
      print 'epoch: %d(%d/%d), iter %d, loss: %f, smooth_loss: %f, pplx: %f' % (cum_p / data_size, cum_p, data_size, n, loss, smooth_loss, pplx)
      dump_filename = 'min_char_rnn_iter%08d_loss%.4f.mat' % (n, smooth_loss)
      sio.savemat(dump_filename, {'Wxh': Wxh, 'Whh': Whh, 'Why': Why, 'bh': bh, 'by': by, 'mWxh': mWxh, 'mWhh': mWhh, 'mWhy': mWhy, 'mbh': mbh, 'mby': mby})
      print 'Dump: %s' % dump_filename
      #clear_output(wait=True)

    p += seq_length # move data pointer
    cum_p += seq_length # cumulate
    n += 1 # iteration counter
