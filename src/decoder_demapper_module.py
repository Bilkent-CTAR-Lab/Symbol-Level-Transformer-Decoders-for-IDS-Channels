import numpy as np
from decoders import LDPC_decoder
from demappers import demapper_2_bit, demapper_3_bit, demapper_4_bit,demapper_5_bit,demapper_6_bit, demapper_8_bit

def decoder_demapper_module_2bit(probs, H, seq, message_length, iter_num = 10, each_iter = 10, abs_limit = 10, early_stop = False):
  """
  Performs joint decoding and demapping for a 2-bit symbol-based communication system using an LDPC decoder.

  Parameters:
  -----------
  probs : np.ndarray
      A 2D array of shape (4, message_length / 2) containing probabilities for each 2-bit symbol. 
      The rows correspond to probabilities for the symbols '00', '01', '10', and '11'.
  H : np.ndarray
      Parity-check matrix for the LDPC decoder.
  seq : np.ndarray
      A 1D binary array of the same length as the message, indicating the scrambling pattern 
      (1 for bit inversion, 0 for no inversion).
  message_length : int
      The total length of the transmitted message in bits.
  iter_num : int, optional
      Number of decoding block iterations to perform (default is 10).
  each_iter : int, optional
      Number of iterations for the LDPC decoder at each decoding stage (default is 10).
  abs_limit : float, optional
      Absolute limit for log-likelihood ratio (LLR) values to avoid numerical overflow (default is 10).
  early_stop : bool, optional
      Whether to stop the LDPC decoder early if a valid codeword is detected (default is False).

  Returns:
  --------
  m_est : np.ndarray
      The estimated message after decoding, represented as a binary array.

  Notes:
  ------
  - The function initializes the LLRs based on the given symbol probabilities and applies LDPC decoding iteratively.
  - The `seq` array is used to handle bit inversions during decoding.
  - The demapping process updates the LLRs after each decoding stage, taking into account both the received symbol probabilities and the output of the LDPC decoder.
  - The function accounts for cases where the message length exceeds the code length (`n`), padding the LLRs as needed.
  
  Example:
  --------
  >>> import numpy as np
  >>> probs = np.random.rand(4, 10)  # Example symbol probabilities
  >>> probs = probs / probs.sum(axis=0, keepdims=True)  # Normalize probabilities
  >>> H = np.array([[1, 0, 1], [0, 1, 1]])  # Example parity-check matrix
  >>> seq = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Example scrambling pattern
  >>> message_length = 10
  >>> m_est = decoder_demapper_module_2bit(probs, H, seq, message_length)
  >>> print(m_est)
  """

  init_llr = np.zeros((1, message_length))
  n = H.shape[-1]
  extra_bits = message_length - n
  # Get initial probs
  for k in range(int(message_length/2)):
    
    # Get corresponding elements
    prob_00 = probs[0, k]
    prob_01 = probs[1, k]
    prob_10 = probs[2, k]
    prob_11 = probs[3, k]

    prob_0 = prob_00 + prob_01
    prob_1 = prob_10 + prob_11
    init_llr[:,2*k] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    prob_0 = prob_00 + prob_10
    prob_1 = prob_01 + prob_11
    init_llr[:,2*k+1] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

  #print(llr < 0)
  #init_llr[seq == 1] = -init_llr[seq == 1]
  #m_est, llrs_from_sum_product = LDPC_decoder(init_llr, H, 100, early_stop = True)
  llr = init_llr

  for i in range(1, iter_num + 1):
    if i < iter_num:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)
      #print(llrs_from_sum_product < 0)
      llrs_from_sum_product = llrs_from_sum_product - llr
      llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1,extra_bits)), axis = 1)
      llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
      llr = demapper_2_bit(probs, llrs_from_sum_product, n=n, message_length = message_length, symbol_bit = 2)
    else:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)

  return m_est

def decoder_demapper_module_3bit(probs, H, seq, message_length, iter_num = 10, each_iter = 10, abs_limit = 10, early_stop = False):

# Inits
  init_llr = np.zeros((1, message_length))
  n = H.shape[-1]
  extra_bits = message_length - n

  for k in range(int(message_length/3)):

    # Get corresponding elements
    prob_000 = probs[0, k]
    prob_001 = probs[1, k]
    prob_010 = probs[2, k]
    prob_011 = probs[3, k]
    prob_100 = probs[4, k]
    prob_101 = probs[5, k]
    prob_110 = probs[6, k]
    prob_111 = probs[7, k]

    prob_0 = prob_000 + prob_001 + prob_010 + prob_011
    prob_1 = prob_100 + prob_101 + prob_110 + prob_111
    init_llr[:,3*k] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    prob_0 = prob_000 + prob_001 + prob_100 + prob_101
    prob_1 = prob_010 + prob_011 + prob_110 + prob_111
    init_llr[:,3*k+1] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    prob_0 = prob_000 + prob_010 + prob_100 + prob_110
    prob_1 = prob_001 + prob_011 + prob_101 + prob_111
    init_llr[:,3*k+2] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

  # As init, input these 
  llr = init_llr

  for i in range(1, iter_num + 1):
    if i < iter_num:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)
      llrs_from_sum_product = llrs_from_sum_product - llr
      llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1,extra_bits)), axis = 1)
      llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
      llr = demapper_3_bit(probs, llrs_from_sum_product, n=n, message_length = message_length, symbol_bit = 3)
    else:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)

  return m_est


def decoder_demapper_module_4bit(probs, H, seq, message_length, iter_num = 10, each_iter = 10, abs_limit = 10, early_stop = False):
  
  # Inits
  init_llr = np.zeros((1, message_length))
  n = H.shape[-1]
  extra_bits = message_length - n

  # Get initial probs
  for k in range(int(message_length/4)):

    # Get corresponding elements
    prob_0000 = probs[0, k]
    prob_0001 = probs[1, k]
    prob_0010 = probs[2, k]
    prob_0011 = probs[3, k]
    prob_0100 = probs[4, k]
    prob_0101 = probs[5, k]
    prob_0110 = probs[6, k]
    prob_0111 = probs[7, k]
    prob_1000 = probs[8, k]
    prob_1001 = probs[9, k]
    prob_1010 = probs[10, k]
    prob_1011 = probs[11, k]
    prob_1100 = probs[12, k]
    prob_1101 = probs[13, k]
    prob_1110 = probs[14, k]
    prob_1111 = probs[15, k]

    prob_0 = prob_0000 + prob_0001 + prob_0010 + prob_0011 + prob_0100 + prob_0101 + prob_0110 + prob_0111
    prob_1 = prob_1000 + prob_1001 + prob_1010 + prob_1011 + prob_1100 + prob_1101 + prob_1110 + prob_1111
    init_llr[:,4*k] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    prob_0 = prob_0000 + prob_0001 + prob_0010 + prob_0011 + prob_1000 + prob_1001 + prob_1010 + prob_1011
    prob_1 = prob_0100 + prob_0101 + prob_0110 + prob_0111 + prob_1100 + prob_1101 + prob_1110 + prob_1111
    init_llr[:,4*k+1] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    prob_0 = prob_0000 + prob_0001 + prob_0100 + prob_0101 + prob_1000 + prob_1001 + prob_1100 + prob_1101
    prob_1 = prob_0010 + prob_0011 + prob_0110 + prob_0111 + prob_1010 + prob_1011 + prob_1110 + prob_1111
    init_llr[:,4*k+2] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    prob_0 = prob_0000 + prob_0010 + prob_0100 + prob_0110 + prob_1000 + prob_1010 + prob_1100 + prob_1110
    prob_1 = prob_0001 + prob_0011 + prob_0101 + prob_0111 + prob_1001 + prob_1011 + prob_1101 + prob_1111
    init_llr[:,4*k+3] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

  # As init, input these 
  llr = init_llr

  for i in range(1, iter_num + 1):
    if i < iter_num:
      #print(i)
      #print(seq.shape)
      #print(llr.shape)
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)
      llrs_from_sum_product = llrs_from_sum_product - llr
      # if message length exceeds n, which can be the case if n does not divide Nc
      llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1,extra_bits)), axis = 1)
      llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
      llr = demapper_4_bit(probs, llrs_from_sum_product, n=n, message_length = message_length, symbol_bit = 4)
    else:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)

  return m_est

def decoder_demapper_module_5bit(probs, H, seq, message_length, iter_num = 10, each_iter = 10, abs_limit = 10, early_stop = False):

  # Inits
  init_llr = np.zeros((1, message_length))
  n = H.shape[-1]
  extra_bits = message_length - n

  for k in range(int(message_length/5)):
    
    set1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,5*k] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    set1 = [0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,5*k+1] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)
    
    set1 = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,5*k+2] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)
    
    set1 =  [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,5*k+3] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)
    
    set1 = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,5*k+4] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

  # As init, input these 
  llr = init_llr

  for i in range(1, iter_num + 1):
    if i < iter_num:
      #print(i)
      #print(seq.shape)
      #print(llr.shape)
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)
      llrs_from_sum_product = llrs_from_sum_product - llr
      llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1,extra_bits)), axis = 1)
      llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
      llr = demapper_5_bit(probs, llrs_from_sum_product, n=n, message_length = message_length, symbol_bit = 5)
    else:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)

  return m_est

def decoder_demapper_module_6bit(probs, H, seq, message_length, iter_num = 10, each_iter = 10, abs_limit = 10, early_stop = False):
  # Inits
  init_llr = np.zeros((1, message_length))
  n = H.shape[-1]
  extra_bits = message_length - n
  #print(H.shape[-1]/4)
  # Get initial probs
  for k in range(int(message_length/6)):
    
    set1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,6*k] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    set1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,6*k+1] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)
    
    set1 = [0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38,39,48,49,50,51,52,53,54,55]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,6*k+2] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)
    
    set1 =  [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,6*k+3] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)
    
    set1 = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,6*k+4] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

    set1 = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62]
    prob_0 = np.sum(probs[set1,k])
    prob_1 = 1-prob_0
    init_llr[:,6*k+5] = np.clip(np.log(prob_0/prob_1), -abs_limit, abs_limit)

  # As init, input these 
  llr = init_llr

  for i in range(1, iter_num + 1):
    if i < iter_num:
      #print(i)
      #print(seq.shape)
      #print(llr.shape)
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)
      llrs_from_sum_product = llrs_from_sum_product - llr
      llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1,extra_bits)), axis = 1)
      llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
      llr = demapper_6_bit(probs, llrs_from_sum_product, n=n, message_length = message_length, symbol_bit = 6)
    else:
      llr[seq == 1] = -llr[seq == 1]
      llr = llr[:,0:n] # if message length exceeds n, which can be the case if n does not divide Nc
      m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop = early_stop)

  return m_est

def decoder_demapper_module_8bit(probs, H, seq, message_length, iter_num=10, each_iter=10, abs_limit=10, early_stop=False):
    """
    Performs joint decoding and demapping for an 8-bit symbol-based communication system using an LDPC decoder.

    Parameters:
    -----------
    probs : np.ndarray
        A 2D array of shape (256, message_length / 8) containing probabilities for each 8-bit symbol.
        Each row corresponds to a specific 8-bit symbol (e.g., 00000000 to 11111111).
    H : np.ndarray
        Parity-check matrix for the LDPC decoder.
    seq : np.ndarray
        A 1D binary array of the same length as the message, indicating the scrambling pattern 
        (1 for bit inversion, 0 for no inversion).
    message_length : int
        The total length of the transmitted message in bits.
    iter_num : int, optional
        Number of decoding block iterations to perform (default is 10).
    each_iter : int, optional
        Number of iterations for the LDPC decoder at each decoding stage (default is 10).
    abs_limit : float, optional
        Absolute limit for log-likelihood ratio (LLR) values to avoid numerical overflow (default is 10).
    early_stop : bool, optional
        Whether to stop the LDPC decoder early if a valid codeword is detected (default is False).

    Returns:
    --------
    m_est : np.ndarray
        The estimated message after decoding, represented as a binary array.
    """
    # Initialize LLRs
    init_llr = np.zeros((1, message_length))
    n = H.shape[-1]
    extra_bits = message_length - n

    # Calculate initial LLRs
    for k in range(int(message_length / 8)):
        for b in range(8):
            bit_indices_0 = [i for i in range(256) if (i >> b) & 1 == 0]
            bit_indices_1 = [i for i in range(256) if (i >> b) & 1 == 1]
            
            prob_0 = np.sum(probs[bit_indices_0, k])
            prob_1 = np.sum(probs[bit_indices_1, k])
            
            init_llr[:, 8 * k + b] = np.clip(np.log(prob_0 / prob_1), -abs_limit, abs_limit)

    # Begin iterative decoding and demapping
    llr = init_llr

    for i in range(1, iter_num + 1):
        if i < iter_num:
            llr[seq == 1] = -llr[seq == 1]
            llr = llr[:, :n]  # Trim if message length exceeds n
            _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop=early_stop)
            llrs_from_sum_product = llrs_from_sum_product - llr
            llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1, extra_bits)), axis=1)
            llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
            llr = demapper_8_bit(probs, llrs_from_sum_product, n=n, message_length=message_length, symbol_bit=8)
        else:
            llr[seq == 1] = -llr[seq == 1]
            llr = llr[:, :n]  # Trim if message length exceeds n
            m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop=early_stop)

    return m_est


def decoder_demapper_module_nbit(probs, H, seq, message_length, symbol_bit, iter_num=10, each_iter=10, abs_limit=10, early_stop=False):
    """
    Generalized joint decoding and demapping for an n-bit symbol-based communication system using an LDPC decoder.

    Parameters:
    -----------
    probs : np.ndarray
        A 2D array of shape (2^symbol_bit, message_length / symbol_bit) containing probabilities for each n-bit symbol.
    H : np.ndarray
        Parity-check matrix for the LDPC decoder.
    seq : np.ndarray
        A 1D binary array of the same length as the message, indicating the scrambling pattern.
    message_length : int
        The total length of the transmitted message in bits.
    symbol_bit : int
        Number of bits per symbol.
    iter_num : int, optional
        Number of decoding block iterations to perform (default is 10).
    each_iter : int, optional
        Number of iterations for the LDPC decoder at each decoding stage (default is 10).
    abs_limit : float, optional
        Absolute limit for log-likelihood ratio (LLR) values to avoid numerical overflow (default is 10).
    early_stop : bool, optional
        Whether to stop the LDPC decoder early if a valid codeword is detected (default is False).

    Returns:
    --------
    m_est : np.ndarray
        The estimated message after decoding, represented as a binary array.
    """
    # Initialization
    init_llr = np.zeros((1, message_length))
    n = H.shape[-1]
    extra_bits = message_length - n
    total_symbols = int(message_length / symbol_bit)

    # Generate bit-mapping sets dynamically
    for k in range(total_symbols):
        for bit in range(symbol_bit):
            indices = [i for i in range(2 ** symbol_bit) if (i >> bit) & 1 == 0]
            prob_0 = np.sum(probs[indices, k])
            prob_1 = 1 - prob_0
            init_llr[:, symbol_bit * k + bit] = np.clip(np.log(prob_0 / prob_1), -abs_limit, abs_limit)

    # Iterative decoding and demapping
    llr = init_llr
    for i in range(1, iter_num + 1):
        if i < iter_num:
            llr[seq == 1] = -llr[seq == 1]
            llr = llr[:, 0:n]  # Truncate to code length if message length exceeds n
            _, llrs_from_sum_product = LDPC_decoder(llr, H, each_iter, early_stop=early_stop)
            llrs_from_sum_product = llrs_from_sum_product - llr
            llrs_from_sum_product = np.append(llrs_from_sum_product, np.zeros((1, extra_bits)), axis=1)
            llrs_from_sum_product[seq == 1] = -llrs_from_sum_product[seq == 1]
            #llr = demapper_generic(probs, llrs_from_sum_product, n=n, message_length=message_length, symbol_bit=symbol_bit)
        else:
            llr[seq == 1] = -llr[seq == 1]
            llr = llr[:, 0:n]
            m_est, _ = LDPC_decoder(llr, H, each_iter, early_stop=early_stop)

    return m_est
