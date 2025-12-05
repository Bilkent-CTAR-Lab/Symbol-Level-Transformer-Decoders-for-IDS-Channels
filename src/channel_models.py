import numpy as np

def ins_del_channel(c, Pd, Pi, Ps):
    """
    Simulates an insertion-deletionchannel with substitution errors.

    Parameters:
    -----------
    c : np.ndarray
        Input codeword, represented as a 2D numpy array where the codeword bits 
        are in the last dimension (e.g., shape = (1, len_c)).
    Pd : float
        Probability of deletion occurring for each bit in the codeword.
    Pi : float
        Probability of insertion occurring for each bit in the codeword.
    Ps : float
        Probability of substitution error for correctly transmitted bits.
    
    Returns:
    --------
    y : list
        Received signal after the channel, represented as a list of bits, 
        potentially including inserted bits and missing bits due to deletions.
    trans : np.ndarray
        Array of transmission outcomes for each bit in the input codeword. 
        Each element is one of:
            'c' : Correct transmission (potentially with substitution error).
            'd' : Deletion of the corresponding input bit.
            'i' : Insertion of extra random bits.
    
    Notes:
    ------
    - The sum of probabilities `Pd + Pi` should not exceed 1, as the 
      remaining probability is assumed to account for correct transmission.
    - The insertion operation generates two random bits for each insertion event.
    - Substitution errors are applied only to bits marked for correct transmission.
    
    Example:
    --------
    >>> import numpy as np
    >>> c = np.array([[1, 0, 1, 1, 0]])
    >>> Pd, Pi, Ps = 0.1, 0.2, 0.05
    >>> y, trans = ins_del_channel(c, Pd, Pi, Ps)
    >>> print(y)
    >>> print(trans)
    """
    len_c = c.shape[-1] # codeword length
    Pt = 1 - Pd - Pi # prob of cor trans

    # get trans probability for each case
    trans = np.random.choice(np.array(['c','d','i']), size = len_c, replace = True, p = [Pt, Pd, Pi])

    # recieved signal
    y = []
    for i in range(len_c):
        # correct transmission
        if trans[i] == 'c':
            rand_m = (c[0,i] + np.random.choice([0,1], size=1, replace=True, p=[1-Ps, Ps])) % 2
            y.append(rand_m.tolist())

        # insertion
        elif trans[i] == 'i':
            i_bit1 = np.random.randint(low = 0, high=2, size=(1), dtype=int)
            y.append(i_bit1.tolist())
            i_bit2 = np.random.randint(low = 0, high=2, size=(1), dtype=int)
            y.append(i_bit2.tolist())

    return y, trans
