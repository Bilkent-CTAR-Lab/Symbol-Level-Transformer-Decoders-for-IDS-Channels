from marker_related import create_codeword_transformer, create_marker_code, create_codeword_bit
from decoder_demapper_module import decoder_demapper_module_2bit, decoder_demapper_module_3bit, decoder_demapper_module_4bit
from decoder_demapper_module import decoder_demapper_module_5bit, decoder_demapper_module_6bit
from channel_models import ins_del_channel
import numpy as np
from tensorflow import keras
import itertools
from decoders import LDPC_decoder

def test_symbol(Pd_array, Pi_array, Ps_array, Nc, marker_code, message_length,n, 
         rate, gamma, symbol_bit, Nr, model, H, temp, mask, 
         iter_num, each_iter, max_codes = 20000, max_fer = 200, 
         abs_limit = 5, print_every = 100):
    
    """Results are in the form =

    (ber, fer, Pd_test_point, Pi_test_point, Ps_test_point, temp_parameter, )

    """

    # Create testing points
    test_points = list(itertools.product(Pd_array,Pi_array,Ps_array))
    # Testing results array
    test_results = []
     # determines total number of symbols + marker symsbols
    symmm = int(message_length/symbol_bit +  message_length/Nc)

    print('TESTING STARTS')
    print("----------------------")
    for i, test_point in enumerate(test_points):

        # Define metrics
        ber_total = 0
        fer_total = 0

        # Get the test point
        Pd_point, Pi_point, Ps_point = test_point
        code_simulated = 0
        print(f"------------Test point (Pd, Pi, Ps) = {test_point}---------------")

        while fer_total <= max_fer and code_simulated <= max_codes:

            np.random.seed(code_simulated)

            # create message vector
            seq = np.random.randint(0,2,(1, message_length))
            #print(seq.shape)
            mes = seq
            # create marker coded bit
            c,_ = create_marker_code(mes, Nc, marker_code)
            y,_ = ins_del_channel(c, Pd_point, Pi_point, Ps_point)
            y = np.array(y).T

            # feed it to nn recursion
            numR = y.shape[-1]
            numT = c.shape[-1]

            if symbol_bit != 1:
                testX = create_codeword_transformer(y, message_length, rate, gamma, numT, symbol_bit, 1, numR, Nc, Nr)
            else:
                testX = create_codeword_bit(y, message_length, rate, gamma, numT, symbol_bit, 1, numR, Nc, Nr)

            probs = model(testX)

            # Temperature Parameter
            probs = probs/temp
            probs = keras.activations.softmax(probs, axis=-1)
            probs = np.array(probs).reshape(symmm,int(2**symbol_bit)).T
            probs = probs[:,mask == 1]
            probs = probs[:,0:n]

            # round it
            if symbol_bit == 2:
                m_est = decoder_demapper_module_2bit(probs = probs, H = H, seq = seq, message_length = message_length,
                                           iter_num = iter_num, each_iter = each_iter,abs_limit = abs_limit, 
                                           early_stop = True)
            elif symbol_bit == 3:
                m_est = decoder_demapper_module_3bit(probs = probs, H = H, seq = seq, message_length = message_length,
                                           iter_num = iter_num, each_iter = each_iter,abs_limit = abs_limit, 
                                           early_stop = True)
            elif symbol_bit == 4:
                m_est = decoder_demapper_module_4bit(probs = probs, H = H, seq = seq, message_length = message_length,
                                           iter_num = iter_num, each_iter = each_iter,abs_limit = abs_limit, 
                                           early_stop = True)
            elif symbol_bit == 5:
                m_est = decoder_demapper_module_5bit(probs = probs, H = H, seq = seq, message_length = message_length,
                                           iter_num = iter_num, each_iter = each_iter,abs_limit = abs_limit, 
                                           early_stop = True)
            elif symbol_bit == 6:
                m_est = decoder_demapper_module_6bit(probs = probs, H = H, seq = seq, message_length = message_length,
                                           iter_num = iter_num, each_iter = each_iter,abs_limit = abs_limit, 
                                           early_stop = True)
            elif symbol_bit == 1:
                probs = probs[0,:].reshape(1,-1)
                llr = np.log((1-probs)/probs)
                llr[seq == 1] = -llr[seq == 1]
                m_est, _ = LDPC_decoder(llr, H, int(each_iter*iter_num), early_stop = True)

            
            # calculate total ber
            ber = np.sum(m_est.astype('int') != np.zeros((1,n)))
            ber_total = ber_total + ber
            code_simulated += 1
            if ber > 0:
                fer_total += 1
            
            if code_simulated % print_every == 0:
                ber = ber_total/(n*code_simulated)
                fer = fer_total/code_simulated
                print(f"{code_simulated}) (Pd, Pi, Ps) = {test_point}, BER: {ber: .7f}, FER: {fer: .7f}")
        
        # Calculate the error rates
        a = ber_total/(code_simulated*n)
        b = fer_total/code_simulated
        # Append the results
        test_results.append((a,b, Pd_point, Pi_point, Ps_point, temp, code_simulated))      
        # Print results at the end of test point
        print(f"{i+1}) (Pd, Pi, Ps) = {test_point}, BER: {a: .7f}, FER: {b: .7f}")

    return test_results

