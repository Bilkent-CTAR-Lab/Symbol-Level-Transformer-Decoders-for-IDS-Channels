import argparse
from test_functions import test_symbol
import scipy
import numpy as np
import sys
from models import BI_LSTM_Insertion_Deletion2
import tensorflow as tf
import os

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--filepath', type=str)
parser.add_argument('--symbol-bit', default=1, type=int, metavar='M')
parser.add_argument('--test-Pd', default=[0.01, 0.015,0.02,0.025,0.03], type=list)
parser.add_argument('--test-Ps', default=[0.01], type=list)
parser.add_argument('--test-Pi', default=[0.000], type=list)
parser.add_argument('--max-codes', default= 10_000, type=int)
parser.add_argument('--max_fer', default = 200, type=int)
parser.add_argument('--each_iter', default = 10, type = int)
parser.add_argument('--iter-num', default = 10, type = int)
parser.add_argument('--temp', default = [1], type = int)
parser.add_argument('--abs-limit', default = 10, type = int)
parser.add_argument('--H', type=int, default=1, help='1 for small, 2 for medium, 3 for large, 4 for xlarge')
parser.add_argument('--Nc', default=12, type=int, metavar='Nc', help='Number of bit to put markers')
parser.add_argument('--print-every', default=100, type=int, metavar='M')
parser.add_argument('--marker', default=np.array([0,1]).reshape(1,-1))
parser.add_argument('--gamma', default=40, type=int,metavar='G')


def test():
    with tf.device('/cpu:0'):
        args = parser.parse_args()
        #if not args.filepath:
            #print("Error: --filepath is required")
            #sys.exit(1)

        # Test arguments
        #filepath = args.filepath
        test_Pd = args.test_Pd
        test_Pi = args.test_Pi
        test_Ps = args.test_Ps
        max_codes = args.max_codes
        max_fer = args.max_fer
        each_iter = args.each_iter
        iter_num = args.iter_num
        abs_limit = args.abs_limit
        temp = args.temp
        Nc = args.Nc
        symbol_bit = args.symbol_bit
        print_every = args.print_every
        marker_code = args.marker
        gamma = args.gamma
        H_choice = args.H
        if H_choice == 1:
            H = scipy.io.loadmat('Matrices/H_matrix_small')['H'] # H matrix
        elif H_choice == 2:
            H = scipy.io.loadmat('Matrices/H_new')['H'] # H matrix
        elif H_choice == 3:
            H = scipy.io.loadmat('Matrices/H_matrix_large')['H'] # H matrix
        elif H_choice == 4:
            H = scipy.io.loadmat('Matrices/H_matrix_xlarge')['H'] # H matrix

        message_length = H.shape[-1] # Total number of message bits
        Nr = marker_code.shape[-1] # Marker sequence length

        if message_length % Nc != 0:
            message_length = message_length + (Nc - (message_length % Nc))
            print("Message length does not divide Nc. New, meesage length is ", message_length)
        else:
            print("Message length divides divide Nc.")

        # Temporary variables
        n = H.shape[-1]
        rate = Nc/(Nc+Nr)
        symmm = int(message_length/Nc + message_length/symbol_bit)
        k = np.arange(int(Nc/symbol_bit), symmm, int(Nc/symbol_bit) + 1)
        mask = np.ones((1,symmm))
        mask[0,k] = 0
        mask = mask.reshape(symmm,)

        model = BI_LSTM_Insertion_Deletion2(d_bilstm = 256, 
                                            d_ffn = [128], 
                                            num_bi_layers = 2, 
                                            output_size = int(2**symbol_bit), 
                                            rnn_type = 'gru')
        zeros = np.zeros((1, symmm, int(2*gamma+1)))
        d = model(zeros)
        filepath = "Results/Medium/symbol_2_gamma_80_nc_20_len_520_epoch_150_size_256_mlp_128_layer_2_lr_0.00090_bs_16_Pd_0.010_Pi_0.000_Ps_0.010_marker_010/symbol_2_gamma_80_nc_20_len_520_epoch_150_size_256_layer_2.h5"
        directory = "Results/Medium/symbol_2_gamma_80_nc_20_len_520_epoch_150_size_256_mlp_128_layer_2_lr_0.00090_bs_16_Pd_0.010_Pi_0.000_Ps_0.010_marker_010"
        model.load_weights(filepath)
        
        directory_open = os.path.join(directory, "test_results")
        if not os.path.exists(directory_open):
            os.mkdir(directory_open)
    
        for i in range(len(temp)):
            test_results = test_symbol(test_Pd, test_Pi, test_Ps, Nc, marker_code, message_length,n, 
                rate, gamma, symbol_bit, Nr, model, H, temp[i], mask, 
                iter_num, each_iter, max_codes = max_codes, max_fer = max_fer, 
                abs_limit = abs_limit, print_every = print_every)
        
            stri = "test_results_temp_%.2f" % (temp[i])
            np.save(os.path.join(directory_open, stri), test_results)
        
if __name__ == '__main__':
    test()