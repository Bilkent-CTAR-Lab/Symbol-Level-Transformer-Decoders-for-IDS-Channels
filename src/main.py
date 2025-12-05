"""
Symbol-Level Training Script for Insertion-Deletion Channel Models
-----------------------------------------------------------------
Trains a Bi-LSTM model for error correction with insertion/deletion noise.
Saves results and model weights for evaluation.

Author: Uras Kargi
License: MIT
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import scipy.io
from tensorflow import keras




from models import Transformer_Insertion_Deletion
from datasets import create_dataset, create_dataset_bitlevel
from test_functions import test_symbol
from functions import save_model_weights, save_args_to_file
from functions import train_step
import logging

parser = argparse.ArgumentParser(description='Training for Symbol Level')
parser.add_argument('--symbol-bit', default=5, type=int, metavar='M')
parser.add_argument('--Nc', default=30, type=int, metavar='Nc', help='Number of bit to put markers')
parser.add_argument('--marker', default=np.array([0,1]).reshape(1,-1))
parser.add_argument('--d-rnn', default = 256, type = int)
parser.add_argument('--d-mlp', default = [64], type = int)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--epochs', default=300, type=int, metavar='epoch', help='Number of total epochs to run')
parser.add_argument('--steps', default=100, type=int, metavar='step', help='Number of steps in each epoch')
parser.add_argument('--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=9e-4, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--gamma', default=40, type=int,metavar='gamma')
parser.add_argument('--train-Pd', default=[0.01, 0.015, 0.02, 0.025], nargs="+")
parser.add_argument('--train-Ps', default=[0.01], nargs="+")
parser.add_argument('--train-Pi', default=[0.00],nargs="+")
parser.add_argument('--wd', '--weight-decay', default=0.75, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', default=1_000_000, type=int, help='seed for initializing training')
parser.add_argument('--H', type=int, default=2 ,help='1 for small, 2 for medium, 3 for large, 4 for xlarge')
parser.add_argument('--main-dir', type=str, default= "./Results")
parser.add_argument('--model-type',
                    type=str,
                    default='transformer',
                    choices=['bilstm', 'transformer'],
                    help='Choose model architecture: "bilstm" or "transformer"')
#parser.add_argument('--test-Pd', default=[0.01, 0.015,0.02,0.025,0.03], type=list)
#parser.add_argument('--test-Ps', default=[0.01], type=list)
#parser.add_argument('--test-Pi', default=[0.0], type=list)
parser.add_argument('--max-codes', default= 20_000, type=int)
parser.add_argument('--max_fer', default = 150, type=int)
parser.add_argument('--each_iter', default = 10, type = int)
parser.add_argument('--iter-num', default = 10, type = int)
parser.add_argument('--temp', default = 1, type = float)
parser.add_argument('--abs-limit', default = 10, type = int)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level (can be DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                   # prints to console
        logging.FileHandler("training.log", mode="w")  # saves to file
    ]
)

def main():
  with tf.device('/cpu:0'):

  #print(tf.version)
    print(keras.__version__)
    print(tf.__version__)
    args = parser.parse_args()

    # Training srguments
    epochs = args.epochs
    steps = args.steps
    batch_size = args.batch_size
    lr = args.lr
    wd = args.wd
    seed = args.seed
    # Model arguments
    gamma = args.gamma
    d_bilstm = args.d_rnn
    mlp_layers = args.d_mlp
    layer_num = args.layers
    main_dir = args.main_dir
    # Marker Code arguments
    symbol_bit = args.symbol_bit
    Nc = args.Nc 
    marker_code = args.marker
    training_Pd = args.train_Pd
    training_Pi = args.train_Pi
    training_Ps = args.train_Ps
    H_choice = args.H
    # Test arguments
    test_Pd =     training_Pd
    test_Pi = training_Pi
    test_Ps = training_Ps
    max_codes = args.max_codes
    max_fer = args.max_fer
    each_iter = args.each_iter
    iter_num = args.iter_num
    abs_limit = args.abs_limit
    temp = args.temp
    
    # Fix Seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Selection of parity check matrix H
    if H_choice == 1:
      H = scipy.io.loadmat('data/Matrices/H_matrix_small')['H'] 
      main_alt_dir = "Small"
    elif H_choice == 2:
      H = scipy.io.loadmat('data/Matrices/H_new')['H']
      main_alt_dir = "Medium"
    elif H_choice == 3:
      H = scipy.io.loadmat('data/Matrices/H_matrix_medium')['H']
      main_alt_dir = "Large"
    elif H_choice == 4:
      H = scipy.io.loadmat('data/Matrices/H_matrix_xlarge')['H']
      main_alt_dir = "X_large"

    # Message lenght 
    message_length = H.shape[-1] # Total number of message bits
    Nr = marker_code.shape[-1] # Marker sequence length
    if message_length % Nc != 0:
        message_length = message_length + (Nc - message_length % Nc)
        logging.info("Message length does not divide Nc. New, meesage length is ", message_length)
    else:
        logging.info("Message length divides divide Nc.")


    symmm = int(message_length/Nc + message_length/symbol_bit)
    k = np.arange(int(Nc/symbol_bit), symmm, int(Nc/symbol_bit) + 1)
    mask = np.ones((1,symmm))
    mask[0,k] = 0
    mask = mask.reshape(symmm,)
    mask_tensor = tf.convert_to_tensor(mask, dtype=tf.int32)
    indices = tf.where(mask_tensor == 1)
    ind = tf.squeeze(indices)
    #print(ind)
    #print(mask_tensor)
    gamma_array = [40, 50, 60, 70, 80, 90, 100]


    if marker_code.shape[-1] == 3:

      
      save_dir = "symbol_%d_gamma_%d_nc_%d_len_%d_epoch_%d_size_%d_mlp_%d_layer_%d_lr_%.5f_bs_%d_Pd_%.3f_Pi_%.3f_Ps_%.3f_marker_%d%d%d" % (symbol_bit, gamma, Nc, 
                                                                                              message_length, 
                                                                                              epochs, d_bilstm, 
                                                                                                  mlp_layers[0],
                                                                                              layer_num, lr,
                                                                                              batch_size,
                                                                                              training_Pd[0],
                                                                                              training_Pi[0],
                                                                                              training_Ps[0],
                                                                                              marker_code[0,0], 
                                                                                              marker_code[0,1], 
                                                                                              marker_code[0,2])
    elif marker_code.shape[-1] == 2:
      save_dir = "symbol_%d_gamma_%d_nc_%d_len_%d_epoch_%d_size_%d_mlp_%d_layer_%d_lr_%.5f_bs_%d_Pd_%.3f_Pi_%.3f_Ps_%.3f_marker_%d%d"  % (symbol_bit, gamma, Nc, 
                                                                                            message_length, 
                                                                                            epochs, d_bilstm, 
                                                                                            mlp_layers[0],
                                                                                            layer_num,lr,
                                                                                            batch_size,
                                                                                            training_Pd[0],
                                                                                            training_Pi[0],
                                                                                            training_Ps[0],
                                                                                            marker_code[0,0], 
                                                                                            marker_code[0,1])
    # Directory creation
    #main_alt_dir_symbol_size = "Symbol_%d"  % (symbol_bit)
    main_dir2 = os.path.join(main_dir, main_alt_dir)
    directory = os.path.join(main_dir2, save_dir)

    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(main_dir2):
        os.mkdir(main_dir2)
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    # Save parameters
    save_args_to_file(args, directory)
    # We use d_rnn as the Transformer d_model for a fair comparison
    model = Transformer_Insertion_Deletion(
        d_model=d_bilstm,
        d_ffn=mlp_layers,
        num_layers=layer_num,
        output_size=int(2**symbol_bit),
        num_heads=12,         # you can tune this if you want
        dropout_rate=0.1
        )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=10000,
                                                                decay_rate=wd,
                                                                staircase = True)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction="sum_over_batch_size",
                                                            from_logits=True, 
                                                            ignore_class = int(2**symbol_bit))

    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)

    train_acc_metric = keras.metrics.BinaryAccuracy()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_acc_metric_topk = keras.metrics.SparseTopKCategoricalAccuracy(k=2)
    model.compile(loss = loss_fn, optimizer = optimizer)

    
    # Train starts
    train_results = []
    print_result = True

    
    for epoch in range(epochs):
        #print("Epoch %d/%d" % (epoch+1, epochs)) # print current epoch
        start_time = time.time() # start time
        train_loss_total = 0
        #check = 0
        train_loss_total = 0

        # Iterate over batches
        for step in range(steps):
          # Create training batch
          if symbol_bit != 1: # If symbol level
            trainX_batch, labels = create_dataset(m_total = message_length, marker_code = marker_code,
                                                  num_code = batch_size, Pd = training_Pd, Pi = training_Pi, 
                                                  Ps = training_Ps, Nc = Nc, symbol_bit = symbol_bit, 
                                                  gamma = gamma)
          elif symbol_bit == 1: # If bit level
            trainX_batch, labels = create_dataset_bitlevel(m_total = message_length, marker_code = marker_code,
                                                          num_code = batch_size, Pd = training_Pd, Pi = training_Pi, 
                                                            Ps = training_Ps, Nc = Nc, gamma = gamma)
      # Record total los
          if symbol_bit != 1:
            train_loss = train_loss = train_step(model, trainX_batch, labels,
                            loss_fn, optimizer,
                            train_acc_metric, train_acc_metric_topk,
                            batch_size=batch_size, symmm=symmm,
                            ind=ind, use_indices=True)
            train_loss_total += train_loss

          else:
            train_loss = train_step(model, trainX_batch, labels,
                            loss_fn, optimizer,
                            train_acc_metric, train_acc_metric_topk,
                            use_indices=False)
            train_loss_total += train_loss 

          train_acc = train_acc_metric.result()
          train_acc_topk = train_acc_metric_topk.result()
          train_loss = train_loss_total/(step + 1)
          #print("Batch: %d - %.2fs - loss: %.4f - acc: %.4f - acc(top k): %.4f" % (step, time.time() - start_time,  train_loss, train_acc, train_acc_topk))
        
        # Get results @ the end of the epoch
        train_acc = train_acc_metric.result()
        train_acc_topk = train_acc_metric_topk.result()
        train_loss = train_loss_total/steps
        train_acc_metric.reset_state()
        train_acc_metric_topk.reset_state()

        if print_result: 
          logging.info("Epoch %d/%d %.2fs - loss: %.4f - acc: %.4f - acc (top k): %.4f" % 
                  (epoch+1, epochs, time.time() - start_time, train_loss, train_acc, train_acc_topk))
          
        # Save results
        train_results.append((train_loss, train_acc, train_acc_topk))

    # Save the results
    np.save(os.path.join(directory, 'train_results'), train_results)
    # Save the model
    model_name = save_dir = "transformer_symbol_%d_gamma_%d_nc_%d_len_%d_epoch_%d_size_%d_layer_%d.weights.h5" % (symbol_bit, gamma, Nc, message_length, epochs, d_bilstm, layer_num)
    save_model_weights(model, path_main = directory,  model_name = model_name)
    
    # Define run length of bi-directional
    
    #perm = np.random.permutation(int(n/symbol_bit))
    #perm = np.arange(int(n/symbol_bit))
    n = H.shape[-1]
    m = H.shape[0]
    k = n - m

    # define deletipn insertion subs probs
    print_every = 1000
    Nr = marker_code.shape[-1]
    rate = Nc/(Nc+Nr)
    
    # BEFORE TEST
    np.random.seed(seed)
    test_results = test_symbol(test_Pd, test_Pi, test_Ps, Nc, marker_code,  message_length, n, 
        rate, gamma, symbol_bit, Nr, model, H, temp, mask, 
        iter_num, each_iter, max_codes = max_codes, max_fer = max_fer, 
        abs_limit = abs_limit, print_every = print_every)


    test_directory = os.path.join(directory, 'test_results')
    stri = "test_results_temp_%.2f" % (temp)
    if not os.path.exists(test_directory):
            os.mkdir(test_directory )
    np.save(os.path.join(test_directory, stri), test_results)

if __name__ == '__main__':
  main()
