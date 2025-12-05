import numpy as np

def demapper_2_bit(probs_from_est, llrs, n, message_length, symbol_bit = 2):
  """
    Demaps 2-bit symbols into bit-level log-likelihood ratios (LLRs) for iterative decoding.

    Parameters:
    -----------
    probs_from_est : np.ndarray
        A 2D array of shape (4, num_symbols), where `num_symbols = message_length / symbol_bit`.
        Represents the symbol-level probabilities for the 2-bit symbols '00', '01', '10', and '11'.
    llrs : np.ndarray
        A 2D array of shape (1, message_length), containing the current bit-level LLRs from an LDPC decoder.
    n : int
        Codeword length (number of bits in the LDPC code).
    message_length : int
        The total length of the transmitted message in bits.
    symbol_bit : int, optional
        Number of bits per symbol (default is 2).

    Returns:
    --------
    new_llrs : np.ndarray
        A 2D array of shape (1, message_length), representing the updated bit-level LLRs.

    Notes:
    ------
    - The function computes updated LLRs for each bit in the message using the provided symbol-level probabilities 
      and the current LLRs from the LDPC decoder.
    - Indicator matrices (`matrix_0_x_2k_1`, `matrix_1_x_2k_1`, `matrix_0_x_2k`, `matrix_1_x_2k`) are used to map 
      symbol probabilities to bit probabilities for each bit in a symbol.
    - The `extra_bits` variable accounts for cases where the message length exceeds the codeword length `n`.
    - Probabilities are calculated in the log domain to prevent numerical underflow.

    Example:
    --------
    >>> import numpy as np
    >>> probs_from_est = np.random.rand(4, 5)  # Example symbol probabilities
    >>> probs_from_est = probs_from_est / probs_from_est.sum(axis=0, keepdims=True)  # Normalize probabilities
    >>> llrs = np.random.randn(1, 10)  # Example LLRs from LDPC
    >>> n = 8  # Codeword length
    >>> message_length = 10
    >>> new_llrs = demapper_2_bit(probs_from_est, llrs, n, message_length)
    >>> print(new_llrs)
    """
  # I expect probs_from_est variable has a shape (4, 102)
  # I expect llrs_from_sum_product has a shape (1, 204)

  # new llrs - output of this demapper
  new_llrs = np.zeros((1,message_length))
  num_symbols = int(message_length/symbol_bit)
  extra_bits = message_length - n

  #Indicator metric for x_(2k-1)
  matrix_0_x_2k_1 = np.array([[1,0], [0,1], [0,0], [0,0]])
  matrix_1_x_2k_1 = np.array([[0,0], [0,0], [1,0], [0,1]])
  #Indicator metric for x_(2k)
  matrix_0_x_2k = np.array([[1,0], [0,0], [0,1], [0,0]])
  matrix_1_x_2k = np.array([[0,0], [1,0], [0,0], [0,1]])

  for k in range(num_symbols):
    # get current probs
    symbol_level_probs = probs_from_est[:,k].reshape(4,1)

    # These are llrs from LDPC directly
    llr_x_2k_1 = llrs[:,2*k]
    # These are derived from the llrs from LDPC
    prob_x_2k_1_1 = 1/(np.exp(llr_x_2k_1) + 1)
    prob_x_2k_1_0 = 1-prob_x_2k_1_1
    prob_x_2k_1 = np.array([prob_x_2k_1_0, prob_x_2k_1_1]).reshape(1,symbol_bit)

    # For bit x_{2k}
    c = np.sum(prob_x_2k_1 * matrix_0_x_2k, axis = -1, keepdims = True)
    d = np.sum(prob_x_2k_1 * matrix_1_x_2k, axis = -1, keepdims = True)
    prob_0 = np.sum(c*symbol_level_probs)
    prob_1 = np.sum(d*symbol_level_probs)
    new_llrs[:,2*k+1] = np.log(prob_0/prob_1)

    llr_x_2k = llrs[:,2*k+1]
    prob_x_2k_1 =  1/(np.exp(llr_x_2k) + 1)
    prob_x_2k_0 = 1- prob_x_2k_1
    prob_x_2k = np.array([prob_x_2k_0, prob_x_2k_1]).reshape(1,symbol_bit)

    # For bit x_{2k-1}
    a = np.sum(prob_x_2k * matrix_0_x_2k_1, axis = -1, keepdims = True)
    b = np.sum(prob_x_2k * matrix_1_x_2k_1, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,2*k] = np.log(prob_0/prob_1)

  new_llrs[:,-extra_bits:]
  return new_llrs

def demapper_3_bit(probs_from_est, llrs, n, message_length, symbol_bit = 3):

  # I expect probs_from_est variable has a shape (4, 102)
  # I expect llrs_from_sum_product has a shape (1, 204)

   # new llrs - output of this demapper
  new_llrs = np.zeros((1,message_length))
  num_symbols = int(message_length/symbol_bit)
  extra_bits = message_length - n

  #Indicator metric for x_(3k-2)
  matrix_0_x_3k_2 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], 
                              [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])
  matrix_1_x_3k_2 = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], 
                              [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
  #Indicator metric for x_(3k-1)
  matrix_0_x_3k_1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,0], [0,0,0,0], 
                              [0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
  matrix_1_x_3k_1 = np.array([[0,0,0,0], [0,0,0,0], [1,0,0,0], [0,1,0,0], 
                              [0,0,0,0], [0,0,0,0], [0,0,1,0], [0,0,0,1]])
  #Indicator metric for x_(3k)
  matrix_0_x_3k_0 = np.array([[1,0,0,0], [0,0,0,0], [0,1,0,0], [0,0,0,0], 
                              [0,0,1,0], [0,0,0,0], [0,0,0,1], [0,0,0,0]])
  matrix_1_x_3k_0 = np.array([[0,0,0,0], [1,0,0,0], [0,0,0,0], [0,1,0,0], 
                              [0,0,0,0], [0,0,1,0], [0,0,0,0], [0,0,0,1]])

  for k in range(num_symbols):
    # get current probs
    symbol_level_probs = probs_from_est[:,k].reshape(8,1)

    # These are llrs from LDPC directly
    llr_x_3k_2 = llrs[:,3*k]
    llr_x_3k_1 = llrs[:,3*k+1]
    llr_x_3k_0 = llrs[:,3*k+2]

    # These are derived from the llrs from LDPC
    prob_x_3k_2_1 = 1/(np.exp(llr_x_3k_2) + 1)
    prob_x_3k_2_0 = 1-prob_x_3k_2_1
    prob_x_3k_1_1 = 1/(np.exp(llr_x_3k_1) + 1)
    prob_x_3k_1_0 = 1-prob_x_3k_1_1
    prob_x_3k_0_1 = 1/(np.exp(llr_x_3k_0) + 1)
    prob_x_3k_0_0 = 1-prob_x_3k_0_1

    p1 = prob_x_3k_1_0*prob_x_3k_0_0
    p2 = prob_x_3k_1_1*prob_x_3k_0_0
    p3 = prob_x_3k_1_0*prob_x_3k_0_1
    p4 = prob_x_3k_1_1*prob_x_3k_0_1
    prob_x_3k_2 = np.array([p1, p3, p2, p4]).reshape(1,4)

    p1 = prob_x_3k_2_0*prob_x_3k_0_0
    p2 = prob_x_3k_2_1*prob_x_3k_0_0
    p3 = prob_x_3k_2_0*prob_x_3k_0_1
    p4 = prob_x_3k_2_1*prob_x_3k_0_1
    prob_x_3k_1 = np.array([p1, p3, p2, p4]).reshape(1,4)

    p1 = prob_x_3k_2_0*prob_x_3k_1_0
    p2 = prob_x_3k_2_1*prob_x_3k_1_0
    p3 = prob_x_3k_2_0*prob_x_3k_1_1
    p4 = prob_x_3k_2_1*prob_x_3k_1_1
    prob_x_3k_0 = np.array([p1, p3, p2, p4]).reshape(1,4)

    # For bit x_{3k-2}
    a = np.sum(prob_x_3k_2 * matrix_0_x_3k_2, axis = -1, keepdims = True)
    b = np.sum(prob_x_3k_2 * matrix_1_x_3k_2, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,3*k] = np.log(prob_0/prob_1)

    # For bit x_{3k-2}
    a = np.sum(prob_x_3k_1 * matrix_0_x_3k_1, axis = -1, keepdims = True)
    b = np.sum(prob_x_3k_1 * matrix_1_x_3k_1, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,3*k+1] = np.log(prob_0/prob_1)

    # For bit x_{3k-2}
    a = np.sum(prob_x_3k_0 * matrix_0_x_3k_0, axis = -1, keepdims = True)
    b = np.sum(prob_x_3k_0 * matrix_1_x_3k_0, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,3*k+2] = np.log(prob_0/prob_1)

  new_llrs[:,-extra_bits:]
  return new_llrs

def demapper_4_bit(probs_from_est, llrs, n, message_length, symbol_bit = 4):

  # I expect probs_from_est variable has a shape (4, 102)
  # I expect llrs_from_sum_product has a shape (1, 204)

  # new llrs - output of this demapper
  new_llrs = np.zeros((1,message_length))
  num_symbols = int(message_length/symbol_bit)
  extra_bits = message_length - n

  #Indicator metric for x_(3k-2)
  matrix_0_x_4k_3 = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], 
                              [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],  
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], 
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]])
  
  matrix_1_x_4k_3 = np.array([[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], 
                              [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],  
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], 
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
  
  matrix_0_x_4k_2 = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], 
                              [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], 
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]])
  
  matrix_1_x_4k_2 = np.array([[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], 
                              [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], 
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
  
  matrix_0_x_4k_1 = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]])
  
  matrix_1_x_4k_1 = np.array([[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
  
  matrix_0_x_4k_0 = np.array([[1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0],  
                              [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0]])
  
  matrix_1_x_4k_0 = np.array([[0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0],  
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0], 
                              [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1]])
  
  for k in range(num_symbols):
    # get current probs
    symbol_level_probs = probs_from_est[:,k].reshape(16,1)

    # These are llrs from LDPC directly
    llr_x_4k_3 = llrs[:,4*k]
    llr_x_4k_2 = llrs[:,4*k+1]
    llr_x_4k_1 = llrs[:,4*k+2]
    llr_x_4k_0 = llrs[:,4*k+3]

    # These are derived from the llrs from LDPC
    prob_x_4k_3_1 = 1/(np.exp(llr_x_4k_3) + 1)
    prob_x_4k_3_0 = 1-prob_x_4k_3_1
    prob_x_4k_2_1 = 1/(np.exp(llr_x_4k_2) + 1)
    prob_x_4k_2_0 = 1-prob_x_4k_2_1
    prob_x_4k_1_1 = 1/(np.exp(llr_x_4k_1) + 1)
    prob_x_4k_1_0 = 1-prob_x_4k_1_1
    prob_x_4k_0_1 = 1/(np.exp(llr_x_4k_0) + 1)
    prob_x_4k_0_0 = 1-prob_x_4k_0_1
    
    p1 = prob_x_4k_0_0*prob_x_4k_1_0*prob_x_4k_2_0 #
    p2 = prob_x_4k_0_0*prob_x_4k_1_0*prob_x_4k_2_1 #
    p3 = prob_x_4k_0_0*prob_x_4k_1_1*prob_x_4k_2_0 #
    p4 = prob_x_4k_0_0*prob_x_4k_1_1*prob_x_4k_2_1
    p5 = prob_x_4k_0_1*prob_x_4k_1_0*prob_x_4k_2_0 #
    p6 = prob_x_4k_0_1*prob_x_4k_1_0*prob_x_4k_2_1
    p7 = prob_x_4k_0_1*prob_x_4k_1_1*prob_x_4k_2_0 #
    p8 = prob_x_4k_0_1*prob_x_4k_1_1*prob_x_4k_2_1 #
    prob_x_4k_3 = np.array([p1,p5,p3,p7,p2,p6,p4,p8]).reshape(1,8)

    p1 = prob_x_4k_0_0*prob_x_4k_1_0*prob_x_4k_3_0
    p2 = prob_x_4k_0_0*prob_x_4k_1_0*prob_x_4k_3_1 
    p3 = prob_x_4k_0_0*prob_x_4k_1_1*prob_x_4k_3_0
    p4 = prob_x_4k_0_0*prob_x_4k_1_1*prob_x_4k_3_1
    p5 = prob_x_4k_0_1*prob_x_4k_1_0*prob_x_4k_3_0
    p6 = prob_x_4k_0_1*prob_x_4k_1_0*prob_x_4k_3_1
    p7 = prob_x_4k_0_1*prob_x_4k_1_1*prob_x_4k_3_0
    p8 = prob_x_4k_0_1*prob_x_4k_1_1*prob_x_4k_3_1
    prob_x_4k_2 = np.array([p1,p5,p3,p7,p2,p6,p4,p8]).reshape(1,8)

    p1 = prob_x_4k_0_0*prob_x_4k_2_0*prob_x_4k_3_0
    p2 = prob_x_4k_0_0*prob_x_4k_2_0*prob_x_4k_3_1 
    p3 = prob_x_4k_0_0*prob_x_4k_2_1*prob_x_4k_3_0
    p4 = prob_x_4k_0_0*prob_x_4k_2_1*prob_x_4k_3_1
    p5 = prob_x_4k_0_1*prob_x_4k_2_0*prob_x_4k_3_0
    p6 = prob_x_4k_0_1*prob_x_4k_2_0*prob_x_4k_3_1
    p7 = prob_x_4k_0_1*prob_x_4k_2_1*prob_x_4k_3_0
    p8 = prob_x_4k_0_1*prob_x_4k_2_1*prob_x_4k_3_1
    prob_x_4k_1 = np.array([p1,p5,p3,p7,p2,p6,p4,p8]).reshape(1,8)

    p1 = prob_x_4k_1_0*prob_x_4k_2_0*prob_x_4k_3_0
    p2 = prob_x_4k_1_0*prob_x_4k_2_0*prob_x_4k_3_1 
    p3 = prob_x_4k_1_0*prob_x_4k_2_1*prob_x_4k_3_0
    p4 = prob_x_4k_1_0*prob_x_4k_2_1*prob_x_4k_3_1
    p5 = prob_x_4k_1_1*prob_x_4k_2_0*prob_x_4k_3_0
    p6 = prob_x_4k_1_1*prob_x_4k_2_0*prob_x_4k_3_1
    p7 = prob_x_4k_1_1*prob_x_4k_2_1*prob_x_4k_3_0
    p8 = prob_x_4k_1_1*prob_x_4k_2_1*prob_x_4k_3_1
    prob_x_4k_0 = np.array([p1,p5,p3,p7,p2,p6,p4,p8]).reshape(1,8)

    a = np.sum(prob_x_4k_3 * matrix_0_x_4k_3, axis = -1, keepdims = True)
    b = np.sum(prob_x_4k_3 * matrix_1_x_4k_3, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,4*k] = np.log(prob_0/prob_1)

    # For bit x_{3k-2}
    a = np.sum(prob_x_4k_2 * matrix_0_x_4k_2, axis = -1, keepdims = True)
    b = np.sum(prob_x_4k_2 * matrix_1_x_4k_2, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,4*k+1] = np.log(prob_0/prob_1)

    # For bit x_{3k-2}
    a = np.sum(prob_x_4k_1 * matrix_0_x_4k_1, axis = -1, keepdims = True)
    b = np.sum(prob_x_4k_1 * matrix_1_x_4k_1, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,4*k+2] = np.log(prob_0/prob_1)

     # For bit x_{3k-2}
    a = np.sum(prob_x_4k_0 * matrix_0_x_4k_0, axis = -1, keepdims = True)
    b = np.sum(prob_x_4k_0 * matrix_1_x_4k_0, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,4*k+3] = np.log(prob_0/prob_1)

  new_llrs[:,-extra_bits:]
  return new_llrs

def demapper_5_bit(probs_from_est, llrs, n, message_length, symbol_bit = 5):

  # new llrs - output of this demapper
  new_llrs = np.zeros((1,message_length))
  num_symbols = int(message_length/symbol_bit)
  extra_bits = message_length - n

  # bases
  other = np.arange(16)
  base = np.arange(32)
  
  #Indicator metric for x_(3k-2)
  set1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_5k_4 = np.zeros((32,16))
  matrix_0_x_5k_4[set1,other] = 1
  matrix_1_x_5k_4 = np.zeros((32,16))
  matrix_1_x_5k_4[set2,other] = 1 
           
  set1 = [0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_5k_3 = np.zeros((32,16))
  matrix_0_x_5k_3[set1,other] = 1
  matrix_1_x_5k_3 = np.zeros((32,16))
  matrix_1_x_5k_3[set2,other] = 1 

  set1 = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_5k_2 = np.zeros((32,16))
  matrix_0_x_5k_2[set1,other] = 1
  matrix_1_x_5k_2 = np.zeros((32,16))
  matrix_1_x_5k_2[set2,other] = 1 

  set1 = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_5k_1 = np.zeros((32,16))
  matrix_0_x_5k_1[set1,other] = 1
  matrix_1_x_5k_1 = np.zeros((32,16))
  matrix_1_x_5k_1[set2,other] = 1 

  set1 = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_5k_0 = np.zeros((32,16))
  matrix_0_x_5k_0[set1,other] = 1
  matrix_1_x_5k_0 = np.zeros((32,16))
  matrix_1_x_5k_0[set2,other] = 1 

  for k in range(num_symbols):
    # get current probs
    symbol_level_probs = probs_from_est[:,k].reshape(32,1)

    # These are llrs from LDPC directly
    llr_x_5k_4 = llrs[:,5*k]
    llr_x_5k_3 = llrs[:,5*k+1]
    llr_x_5k_2 = llrs[:,5*k+2]
    llr_x_5k_1 = llrs[:,5*k+3]
    llr_x_5k_0 = llrs[:,5*k+4]

    # These are derived from the llrs from LDPC
    prob_x_5k_4_1 = 1/(np.exp(llr_x_5k_4) + 1)
    prob_x_5k_4_0 = 1-prob_x_5k_4_1
    prob_x_5k_3_1 = 1/(np.exp(llr_x_5k_3) + 1)
    prob_x_5k_3_0 = 1-prob_x_5k_3_1
    prob_x_5k_2_1 = 1/(np.exp(llr_x_5k_2) + 1)
    prob_x_5k_2_0 = 1-prob_x_5k_2_1
    prob_x_5k_1_1 = 1/(np.exp(llr_x_5k_1) + 1)
    prob_x_5k_1_0 = 1-prob_x_5k_1_1
    prob_x_5k_0_1 = 1/(np.exp(llr_x_5k_0) + 1)
    prob_x_5k_0_0 = 1-prob_x_5k_0_1
    

    p1  = prob_x_5k_3_0*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_0
    p2  = prob_x_5k_3_0*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_1
    p3  = prob_x_5k_3_0*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_0
    p4  = prob_x_5k_3_0*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_1
    p5  = prob_x_5k_3_0*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_0
    p6  = prob_x_5k_3_0*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_1
    p7  = prob_x_5k_3_0*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_0
    p8  = prob_x_5k_3_0*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_1
    p9  = prob_x_5k_3_1*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_0
    p10 = prob_x_5k_3_1*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_1
    p11 = prob_x_5k_3_1*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_0
    p12 = prob_x_5k_3_1*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_1
    p13 = prob_x_5k_3_1*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_0
    p14 = prob_x_5k_3_1*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_1
    p15 = prob_x_5k_3_1*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_0
    p16 = prob_x_5k_3_1*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_1
    prob_x_5k_4 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16]).reshape(1,16)

    p1  = prob_x_5k_4_0*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_0
    p2  = prob_x_5k_4_0*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_1
    p3  = prob_x_5k_4_0*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_0
    p4  = prob_x_5k_4_0*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_1
    p5  = prob_x_5k_4_0*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_0
    p6  = prob_x_5k_4_0*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_1
    p7  = prob_x_5k_4_0*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_0
    p8  = prob_x_5k_4_0*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_1
    p9  = prob_x_5k_4_1*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_0
    p10 = prob_x_5k_4_1*prob_x_5k_2_0*prob_x_5k_1_0 *prob_x_5k_0_1
    p11 = prob_x_5k_4_1*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_0
    p12 = prob_x_5k_4_1*prob_x_5k_2_0*prob_x_5k_1_1 *prob_x_5k_0_1
    p13 = prob_x_5k_4_1*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_0
    p14 = prob_x_5k_4_1*prob_x_5k_2_1*prob_x_5k_1_0 *prob_x_5k_0_1
    p15 = prob_x_5k_4_1*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_0
    p16 = prob_x_5k_4_1*prob_x_5k_2_1*prob_x_5k_1_1 *prob_x_5k_0_1
    prob_x_5k_3 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16]).reshape(1,16)

    p1  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_1_0 *prob_x_5k_0_0
    p2  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_1_0 *prob_x_5k_0_1
    p3  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_1_1 *prob_x_5k_0_0
    p4  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_1_1 *prob_x_5k_0_1
    p5  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_1_0 *prob_x_5k_0_0
    p6  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_1_0 *prob_x_5k_0_1
    p7  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_1_1 *prob_x_5k_0_0
    p8  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_1_1 *prob_x_5k_0_1
    p9  = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_1_0 *prob_x_5k_0_0
    p10 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_1_0 *prob_x_5k_0_1
    p11 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_1_1 *prob_x_5k_0_0
    p12 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_1_1 *prob_x_5k_0_1
    p13 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_1_0 *prob_x_5k_0_0
    p14 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_1_0 *prob_x_5k_0_1
    p15 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_1_1 *prob_x_5k_0_0
    p16 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_1_1 *prob_x_5k_0_1
    prob_x_5k_2 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16]).reshape(1,16)

    p1  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_0_0
    p2  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_0_1
    p3  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_0_0
    p4  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_0_1
    p5  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_0_0
    p6  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_0_1
    p7  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_0_0
    p8  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_0_1
    p9  = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_0_0
    p10 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_0_1
    p11 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_0_0
    p12 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_0_1
    p13 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_0_0
    p14 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_0_1
    p15 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_0_0
    p16 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_0_1
    prob_x_5k_1 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16]).reshape(1,16)
    
    p1  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_1_0
    p2  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_1_1
    p3  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_1_0
    p4  = prob_x_5k_4_0*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_1_1
    p5  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_1_0
    p6  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_1_1
    p7  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_1_0
    p8  = prob_x_5k_4_0*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_1_1
    p9  = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_1_0
    p10 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_0 *prob_x_5k_1_1
    p11 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_1_0
    p12 = prob_x_5k_4_1*prob_x_5k_3_0*prob_x_5k_2_1 *prob_x_5k_1_1
    p13 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_1_0
    p14 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_0 *prob_x_5k_1_1
    p15 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_1_0
    p16 = prob_x_5k_4_1*prob_x_5k_3_1*prob_x_5k_2_1 *prob_x_5k_1_1
    prob_x_5k_0 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16]).reshape(1,16)


    a = np.sum(prob_x_5k_4 * matrix_0_x_5k_4, axis = -1, keepdims = True)
    b = np.sum(prob_x_5k_4 * matrix_1_x_5k_4, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,5*k] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_5k_3 * matrix_0_x_5k_3, axis = -1, keepdims = True)
    b = np.sum(prob_x_5k_3 * matrix_1_x_5k_3, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,5*k+1] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_5k_2 * matrix_0_x_5k_2, axis = -1, keepdims = True)
    b = np.sum(prob_x_5k_2 * matrix_1_x_5k_2, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,5*k+2] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_5k_1 * matrix_0_x_5k_1, axis = -1, keepdims = True)
    b = np.sum(prob_x_5k_1 * matrix_1_x_5k_1, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,5*k+3] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_5k_0 * matrix_0_x_5k_0, axis = -1, keepdims = True)
    b = np.sum(prob_x_5k_0 * matrix_1_x_5k_0, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,5*k+4] = np.log(prob_0/prob_1)

  new_llrs[:,-extra_bits:]
  return new_llrs

def demapper_6_bit(probs_from_est, llrs, n, message_length, symbol_bit = 6):

  # new llrs - output of this demapper
  new_llrs = np.zeros((1,message_length))
  num_symbols = int(message_length/symbol_bit)
  extra_bits = message_length - n

  # bases
  other = np.arange(32)
  base = np.arange(64)
  n1 = 32
  n2 = 64
  
  #Indicator metric for x_(3k-2)
  set1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_6k_5 = np.zeros((n2,n1))
  matrix_0_x_6k_5[set1,other] = 1
  matrix_1_x_6k_5 = np.zeros((n2,n1))
  matrix_1_x_6k_5[set2,other] = 1 
           
  set1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_6k_4 = np.zeros((n2,n1))
  matrix_0_x_6k_4[set1,other] = 1
  matrix_1_x_6k_4 = np.zeros((n2,n1))
  matrix_1_x_6k_4[set2,other] = 1 

  set1 = [0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38,39,48,49,50,51,52,53,54,55]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_6k_3 =np.zeros((n2,n1))
  matrix_0_x_6k_3[set1,other] = 1
  matrix_1_x_6k_3 = np.zeros((n2,n1))
  matrix_1_x_6k_3[set2,other] = 1 

  set1 =  [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_6k_2 = np.zeros((n2,n1))
  matrix_0_x_6k_2[set1,other] = 1
  matrix_1_x_6k_2 = np.zeros((n2,n1))
  matrix_1_x_6k_2[set2,other] = 1 

  set1 = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_6k_1 = np.zeros((n2,n1))
  matrix_0_x_6k_1[set1,other] = 1
  matrix_1_x_6k_1 = np.zeros((n2,n1))
  matrix_1_x_6k_1[set2,other] = 1 

  set1 = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62]
  set2 = np.setdiff1d(base, set1)
  matrix_0_x_6k_0 = np.zeros((n2,n1))
  matrix_0_x_6k_0[set1,other] = 1
  matrix_1_x_6k_0 = np.zeros((n2,n1))
  matrix_1_x_6k_0[set2,other] = 1 

  for k in range(num_symbols):
    # get current probs
    symbol_level_probs = probs_from_est[:,k].reshape(64,1)

    # These are llrs from LDPC directly
    llr_x_6k_5 = llrs[:,6*k]
    llr_x_6k_4 = llrs[:,6*k+1]
    llr_x_6k_3 = llrs[:,6*k+2]
    llr_x_6k_2 = llrs[:,6*k+3]
    llr_x_6k_1 = llrs[:,6*k+4]
    llr_x_6k_0 = llrs[:,6*k+5]

    # These are derived from the llrs from LDPC
    prob_x_6k_5_1 = 1/(np.exp(llr_x_6k_5) + 1)
    prob_x_6k_5_0 = 1-prob_x_6k_5_1
    prob_x_6k_4_1 = 1/(np.exp(llr_x_6k_4) + 1)
    prob_x_6k_4_0 = 1-prob_x_6k_4_1
    prob_x_6k_3_1 = 1/(np.exp(llr_x_6k_3) + 1)
    prob_x_6k_3_0 = 1-prob_x_6k_3_1
    prob_x_6k_2_1 = 1/(np.exp(llr_x_6k_2) + 1)
    prob_x_6k_2_0 = 1-prob_x_6k_2_1
    prob_x_6k_1_1 = 1/(np.exp(llr_x_6k_1) + 1)
    prob_x_6k_1_0 = 1-prob_x_6k_1_1
    prob_x_6k_0_1 = 1/(np.exp(llr_x_6k_0) + 1)
    prob_x_6k_0_0 = 1-prob_x_6k_0_1
    

    p1  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p2  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p3  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p4  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p5  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p6  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p7  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p8  = prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p9  = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p10 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p11 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p12 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p13 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p14 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p15 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p16 = prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p17  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p18  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p19  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p20  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p21  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p22  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p23  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p24  = prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p25  = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p26 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p27 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p28 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p29 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p30 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p31 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p32 = prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    prob_x_6k_5 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16,
                            p17,p18,p19,p20,p21,p22,p23,p24,
                            p25,p26,p27,p28,p29,p30,p31,p32]).reshape(1,32)

    p1  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p2  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p3  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p4  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p5  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p6  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p7  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p8  = prob_x_6k_5_0*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p9  = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p10 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p11 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p12 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p13 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p14 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p15 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p16 = prob_x_6k_5_0*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p17  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p18  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p19  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p20  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p21  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p22  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p23  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p24  = prob_x_6k_5_1*prob_x_6k_3_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p25  = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p26 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p27 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p28 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p29 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p30 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p31 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p32 = prob_x_6k_5_1*prob_x_6k_3_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    prob_x_6k_4 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16,
                            p17,p18,p19,p20,p21,p22,p23,p24,
                            p25,p26,p27,p28,p29,p30,p31,p32]).reshape(1,32)

    p1  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p2  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p3  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p4  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p5  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p6  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p7  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p8  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p9  = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p10 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p11 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p12 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p13 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p14 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p15 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p16 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p17  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p18  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p19  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p20  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p21  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p22  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p23  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p24  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p25  = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p26 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p27 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p28 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p29 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p30 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p31 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p32 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_2_1*prob_x_6k_1_1 *prob_x_6k_0_1
    prob_x_6k_3 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16,
                            p17,p18,p19,p20,p21,p22,p23,p24,
                            p25,p26,p27,p28,p29,p30,p31,p32]).reshape(1,32)

    p1  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p2  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p3  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p4  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p5  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p6  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p7  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p8  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p9  = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p10 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p11 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p12 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p13 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p14 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p15 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p16 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p17  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p18  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p19  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p20  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p21  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p22  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p23  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p24  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_1
    p25  = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_0
    p26 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_0 *prob_x_6k_0_1
    p27 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_0
    p28 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_1_1 *prob_x_6k_0_1
    p29 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_0
    p30 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_0 *prob_x_6k_0_1
    p31 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_0
    p32 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_1_1 *prob_x_6k_0_1
    prob_x_6k_2 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16,
                            p17,p18,p19,p20,p21,p22,p23,p24,
                            p25,p26,p27,p28,p29,p30,p31,p32]).reshape(1,32)
    
    p1  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_0
    p2  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_1
    p3  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_0
    p4  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_1
    p5  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_0
    p6  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_1
    p7  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_0
    p8  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_1
    p9  = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_0
    p10 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_1
    p11 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_0
    p12 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_1
    p13 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_0
    p14 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_1
    p15 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_0
    p16 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_1
    p17  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_0
    p18  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_1
    p19  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_0
    p20  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_1
    p21  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_0
    p22  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_1
    p23  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_0
    p24  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_1
    p25  = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_0
    p26 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_0_1
    p27 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_0
    p28 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_0_1
    p29 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_0
    p30 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_0_1
    p31 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_0
    p32 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_0_1
    prob_x_6k_1 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16,
                            p17,p18,p19,p20,p21,p22,p23,p24,
                            p25,p26,p27,p28,p29,p30,p31,p32]).reshape(1,32)
    
    p1  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_0
    p2  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_1
    p3  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_0
    p4  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_1
    p5  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_0
    p6  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_1
    p7  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_0
    p8  = prob_x_6k_5_0*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_1
    p9  = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_0
    p10 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_1
    p11 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_0
    p12 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_1
    p13 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_0
    p14 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_1
    p15 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_0
    p16 = prob_x_6k_5_0*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_1
    p17  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_0
    p18  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_1
    p19  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_0
    p20  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_1
    p21  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_0
    p22  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_1
    p23  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_0
    p24  = prob_x_6k_5_1*prob_x_6k_4_0*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_1
    p25  = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_0
    p26 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_0 *prob_x_6k_1_1
    p27 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_0
    p28 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_0*prob_x_6k_2_1 *prob_x_6k_1_1
    p29 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_0
    p30 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_0 *prob_x_6k_1_1
    p31 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_0
    p32 = prob_x_6k_5_1*prob_x_6k_4_1*prob_x_6k_3_1*prob_x_6k_2_1 *prob_x_6k_1_1
    prob_x_6k_0 = np.array([p1,p2,p3,p4,p5,p6,p7,p8,
                            p9,p10,p11,p12,p13,p14,p15,p16,
                            p17,p18,p19,p20,p21,p22,p23,p24,
                            p25,p26,p27,p28,p29,p30,p31,p32]).reshape(1,32)


    a = np.sum(prob_x_6k_5 * matrix_0_x_6k_5, axis = -1, keepdims = True)
    b = np.sum(prob_x_6k_5 * matrix_1_x_6k_5, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,6*k] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_6k_4 * matrix_0_x_6k_4, axis = -1, keepdims = True)
    b = np.sum(prob_x_6k_4 * matrix_1_x_6k_4, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,6*k+1] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_6k_3 * matrix_0_x_6k_3, axis = -1, keepdims = True)
    b = np.sum(prob_x_6k_3 * matrix_1_x_6k_3, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,6*k+2] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_6k_2 * matrix_0_x_6k_2, axis = -1, keepdims = True)
    b = np.sum(prob_x_6k_2 * matrix_1_x_6k_2, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,6*k+3] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_6k_1 * matrix_0_x_6k_1, axis = -1, keepdims = True)
    b = np.sum(prob_x_6k_1 * matrix_1_x_6k_1, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,6*k+4] = np.log(prob_0/prob_1)

    a = np.sum(prob_x_6k_0 * matrix_0_x_6k_0, axis = -1, keepdims = True)
    b = np.sum(prob_x_6k_0 * matrix_1_x_6k_0, axis = -1, keepdims = True)
    prob_0 = np.sum(a*symbol_level_probs)
    prob_1 = np.sum(b*symbol_level_probs)
    new_llrs[:,6*k+5] = np.log(prob_0/prob_1)

  new_llrs[:,-extra_bits:]
  return new_llrs

def demapper_8_bit(probs_from_est, llrs, n, message_length, symbol_bit=8):
    """
    Demaps 8-bit symbols into bit-level log-likelihood ratios (LLRs) for iterative decoding.

    Parameters:
    -----------
    probs_from_est : np.ndarray
        A 2D array of shape (256, num_symbols), where num_symbols = message_length / symbol_bit.
        Represents the symbol-level probabilities for the 8-bit symbols.
    llrs : np.ndarray
        A 2D array of shape (1, message_length), containing the current bit-level LLRs from an LDPC decoder.
    n : int
        Codeword length (number of bits in the LDPC code).
    message_length : int
        The total length of the transmitted message in bits.
    symbol_bit : int, optional
        Number of bits per symbol (default is 8).

    Returns:
    --------
    new_llrs : np.ndarray
        A 2D array of shape (1, message_length), representing the updated bit-level LLRs.
    """

    new_llrs = np.zeros((1, message_length))
    num_symbols = int(message_length / symbol_bit)
    extra_bits = message_length - n

    indicator_matrices = []
    for bit_pos in range(symbol_bit):
        matrix_0 = np.zeros((2**symbol_bit, 2**(symbol_bit - bit_pos - 1)))
        matrix_1 = np.zeros((2**symbol_bit, 2**(symbol_bit - bit_pos - 1)))

        for idx in range(2**symbol_bit):
            binary_repr = format(idx, f'0{symbol_bit}b')
            bit_value = int(binary_repr[bit_pos])

            if bit_value == 0:
                matrix_0[idx, idx % 2**(symbol_bit - bit_pos - 1)] = 1
            else:
                matrix_1[idx, idx % 2**(symbol_bit - bit_pos - 1)] = 1

        indicator_matrices.append((matrix_0, matrix_1))

    for k in range(num_symbols):
        symbol_level_probs = probs_from_est[:, k].reshape(-1, 1)

        for bit_pos in range(symbol_bit):
            llr = llrs[:, k * symbol_bit + bit_pos]
            prob_1 = 1 / (np.exp(llr) + 1)
            prob_0 = 1 - prob_1
            prob = np.array([prob_0, prob_1]).reshape(1, 2)

            matrix_0, matrix_1 = indicator_matrices[bit_pos]
            prob_0_combined = np.sum(prob * matrix_0, axis=-1, keepdims=True)
            prob_1_combined = np.sum(prob * matrix_1, axis=-1, keepdims=True)

            bit_prob_0 = np.sum(prob_0_combined * symbol_level_probs)
            bit_prob_1 = np.sum(prob_1_combined * symbol_level_probs)

            new_llrs[:, k * symbol_bit + bit_pos] = np.log(bit_prob_0 / bit_prob_1)

    new_llrs[:, -extra_bits:]  # Preserve any extra bits at the end.
    return new_llrs


def demapper_n_bit(probs_from_est, llrs, n, message_length, symbol_bit):
    """
    Generalized demapper function for n-bit symbols.
    
    Parameters:
    -----------
    probs_from_est : np.ndarray
        A 2D array of shape (2**symbol_bit, num_symbols), where `num_symbols = message_length / symbol_bit`.
        Represents the symbol-level probabilities for n-bit symbols.
    llrs : np.ndarray
        A 2D array of shape (1, message_length), containing the current bit-level LLRs from an LDPC decoder.
    n : int
        Codeword length (number of bits in the LDPC code).
    message_length : int
        The total length of the transmitted message in bits.
    symbol_bit : int
        Number of bits per symbol.

    Returns:
    --------
    new_llrs : np.ndarray
        A 2D array of shape (1, message_length), representing the updated bit-level LLRs.
    """
    new_llrs = np.zeros((1, message_length))
    num_symbols = int(message_length / symbol_bit)
    extra_bits = message_length - n

    # Precompute indicator matrices
    bit_patterns = np.array([[int(bit) for bit in format(i, f'0{symbol_bit}b')] for i in range(2**symbol_bit)])
    indicator_matrices_0 = [np.zeros((2**symbol_bit, 2**(symbol_bit - 1))) for _ in range(symbol_bit)]
    indicator_matrices_1 = [np.zeros((2**symbol_bit, 2**(symbol_bit - 1))) for _ in range(symbol_bit)]
    
    for bit_idx in range(symbol_bit):
        for sym_idx, pattern in enumerate(bit_patterns):
            row_group = sym_idx // (2**(symbol_bit - bit_idx - 1))
            col_group = sym_idx % (2**(symbol_bit - bit_idx - 1))
            if pattern[bit_idx] == 0:
                indicator_matrices_0[bit_idx][sym_idx, col_group] = 1
            else:
                indicator_matrices_1[bit_idx][sym_idx, col_group] = 1

    # Process each symbol
    for k in range(num_symbols):
        symbol_level_probs = probs_from_est[:, k].reshape(-1, 1)

        # Compute bit probabilities
        bit_probabilities = []
        for bit_idx in range(symbol_bit):
            prob_1 = 1 / (np.exp(llrs[:, symbol_bit * k + bit_idx]) + 1)
            prob_0 = 1 - prob_1
            bit_probabilities.append(np.array([prob_0, prob_1]).reshape(1, 2))
        
        for bit_idx in range(symbol_bit):
            # Combine probabilities for the current bit
            combined_probs = np.prod([
                bit_probabilities[b][0, 0 if b == bit_idx else 1] if b != bit_idx else bit_probabilities[b][0, :]
                for b in range(symbol_bit)
            ], axis=0).reshape(1, -1)
            
            # Compute new LLR for the current bit
            prob_0 = np.sum(indicator_matrices_0[bit_idx] @ combined_probs.T * symbol_level_probs)
            prob_1 = np.sum(indicator_matrices_1[bit_idx] @ combined_probs.T * symbol_level_probs)
            new_llrs[:, symbol_bit * k + bit_idx] = np.log(prob_0 / prob_1)

    # Handle extra bits (if any)
    if extra_bits > 0:
        new_llrs[:, -extra_bits:] = llrs[:, -extra_bits:]

    return new_llrs
