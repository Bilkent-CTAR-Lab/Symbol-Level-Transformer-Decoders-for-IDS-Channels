import numpy as np
from functions import bits_to_int
from channel_models import ins_del_channel
from marker_related import create_marker_code


def create_dataset_bitlevel(m_total, marker_code = np.array([0,1]).reshape(1,-1), num_code = 100,
                                  Pd = [0.05], Pi = [0.00], Ps = [0.00], Nc = 10, gamma = 20):

    def is_special_index(j):
        return any((j + 1) % (Nc + k) == 0 for k in range(1, Nr + 1))
    # Marker Code and its parameters
    Nr = marker_code.shape[-1]
    # Rate of the inner code
    r = Nc/(Nc+Nr)
    # total number of symbols
    total_symbols = 2

    # If shape does not divide N_c
    if m_total % Nc != 0:
        m_total = m_total + (Nc - (m_total % Nc))

    symmm = int(m_total/r)
    trainX = np.zeros((num_code, symmm, int(2*gamma+1))).astype(float)
    trainY = np.zeros((num_code, symmm, 1)).astype(int)
    
    for i in range(num_code):
      # Sample from probs
      Pd_sample = np.random.choice(Pd)
      Ps_sample = np.random.choice(Ps)
      Pi_sample = np.random.choice(Pi)

      # Create a random message
      m = np.random.randint(0,2,size = (1, m_total))
      # Create a marker coded codeword
      c,_ = create_marker_code(m, Nc, marker_code)
      # Create a channel realization by sending the codeword through channel and shape it
      y,_ = ins_del_channel(c, Pd_sample, Pi_sample, Ps_sample)
      y = np.array(y).T
      numR = y.shape[-1]
      #numT = c.shape[-1]
      a = 0
      for j in range(symmm):
        if is_special_index(j):
          if a <= gamma:
            trainX[i,j,0:gamma-a] =  0
            trainX[i,j,gamma-a:] = -2*y[0, 0:gamma+a+1] + 1
          elif gamma < a and a < numR-gamma:
            trainX[i,j,:] = -2*y[0, a-gamma:a+gamma+1] + 1
          elif numR-gamma <= a and a < numR + gamma:
            trainX[i,j,0:numR-a+gamma] = -2*y[0, a-gamma:numR] + 1
            trainX[i,j,numR-a+gamma:] = 0
          a += 1
        else:
          a += 1

      a = 0
      for j in range(symmm):
        if not is_special_index(j):
          trainY[i,j,0] = c[0,a]
          a += 1
        else:
          trainY[i,j,0] = total_symbols
          a += 1

    return trainX, trainY

def create_codeword_bit(y, n, r, gamma, numT, symbol_bit, approach, numR, Nc, Nr):
  def is_special_index(j):
    return any((j + 1) % (Nc + k) == 0 for k in range(1, Nr + 1))
   
  symmm = int(n/r)
  trainX = np.zeros((1, symmm, int(2*gamma+1))).astype(float)
  #trainY = np.zeros((1, int(n/r/symbol_bit), 1)).astype(int)

  a = 0
  for j in range(symmm):
        if is_special_index(j):
          if a <= gamma:
            trainX[0,j,0:gamma-a] =  0
            trainX[0,j,gamma-a:] = -2*y[0, 0:gamma+a+1] + 1
          elif gamma < a and a < numR-gamma:
            trainX[0,j,:] = -2*y[0, a-gamma:a+gamma+1] + 1
          elif numR-gamma <= a and a < numR + gamma:
            trainX[0,j,0:numR-a+gamma] = -2*y[0, a-gamma:numR] + 1
            trainX[0,j,numR-a+gamma:] = 0
          a += 1
        else:
          a += 1

  return trainX


def create_dataset_transformer(m_total, marker_code = np.array([0,1]).reshape(1,-1), num_code = 100,
                                  Pd = [0.05], Pi = [0.00], Ps = [0.00], Nc = 10, symbol_bit = 3, gamma = 20):

    # Marker Code and its parameters
    Nr = marker_code.shape[-1]
    # Rate of the inner code
    #r = Nc/(Nc+Nr)
    # total number of symbols
    total_symbols = int(2**symbol_bit)

    assert Nc % symbol_bit == 0, "Nc should divide symbol bit"

    # If shape does not divide N_c
    if m_total % Nc != 0:
        m_total = m_total + (Nc - (m_total % Nc))

    symmm = int(m_total/symbol_bit + m_total/Nc)
    trainX = np.zeros((num_code, symmm, int(2*gamma+1))).astype(float)
    trainY = np.zeros((num_code, symmm, 1)).astype(int)
    
    for i in range(num_code):
      # Sample from probs
      Pd_sample = np.random.choice(Pd)
      Ps_sample = np.random.choice(Ps)
      Pi_sample = np.random.choice(Pi)

      # Create a random message
      m = np.random.randint(0,2,size = (1, m_total))
      # Create a marker coded codeword
      c,_ = create_marker_code(m, Nc, marker_code)
      # Create a channel realization by sending the codeword through channel and shape it
      y,_ = ins_del_channel(c, Pd_sample, Pi_sample, Ps_sample)
      y = np.array(y).T
      numR = y.shape[-1]
      #numT = c.shape[-1]
      a = 0
      for j in range(symmm):
        if (j + 1) % (int(Nc/symbol_bit) + 1) == 0:
          if a <= gamma:
            trainX[i,j,0:gamma-a] =  0
            trainX[i,j,gamma-a:] = -2*y[0, 0:gamma+a+1] + 1
          elif gamma < a and a < numR-gamma:
            trainX[i,j,:] = -2*y[0, a-gamma:a+gamma+1] + 1
          elif numR-gamma <= a and a < numR + gamma:
            trainX[i,j,0:numR-a+gamma] = -2*y[0, a-gamma:numR] + 1
            trainX[i,j,numR-a+gamma:] = 0
          a += Nr
        else:
          a += symbol_bit

      a = 0
      for j in range(symmm):
        if (j + 1) %  (int(Nc/symbol_bit) + 1) != 0:
          trainY[i, j,0] = int(bits_to_int(symbol_bit, c[0,a:a+symbol_bit].astype(int)))
          a += symbol_bit
        else:
          trainY[i, j,0] = total_symbols
          a += Nr

    return trainX, trainY





