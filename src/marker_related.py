import numpy as np
#from functions import bits_to_int
#from channel_models import ins_del_channel

def create_marker_code(m, Nc, marker_code):
    # parameters of the code
    Nr = marker_code.shape[-1]
    N = Nr + Nc
    rm = Nc/(Nc+Nr)

    if m.shape[-1] % Nc != 0:
        m = np.concatenate((m, np.zeros((1, Nc-(m.shape[-1] % Nc)))), axis = 1) 

    mtotal = m.shape[-1]
    # create marker coded bit
    c = np.zeros((1, int(mtotal/rm)))

    # marker bit
    mask = np.zeros((1, int(mtotal/rm)))
    for i in range(int(mtotal/Nc)):
        low_ind = N*i # low ind
        high_ind = N*(i+1)# high ind
        low_ind_m = (N-Nr)*i
        high_ind_m = (N-Nr)*(i+1)

        c[0, low_ind : high_ind - Nr] = m[0, low_ind_m: high_ind_m]
        c[0, high_ind - Nr: high_ind] = marker_code
        mask[0, high_ind - Nr: high_ind] = np.ones((1, Nr))

    return c, mask

def create_codeword_transformer(y, n, r, gamma, numT, symbol_bit, approach, numR, Nc, Nr):

    symmm = int(n/symbol_bit + n/Nc)
    trainX = np.zeros((1, symmm, int(2*gamma+1))).astype(float)
    #trainY = np.zeros((1, int(n/r/symbol_bit), 1)).astype(int)

    approach = 0
    if approach == 0:
      for j in range(symmm):
        if symbol_bit*j <= gamma:
          trainX[0,j,0:gamma-symbol_bit*j] =  0
          trainX[0,j,gamma-symbol_bit*j:] = -2*y[0, 0:gamma+symbol_bit*j+1] + 1
        elif gamma < symbol_bit*j and symbol_bit*j < numR-gamma:
          trainX[0,j,:] = -2*y[0, symbol_bit*j-gamma:symbol_bit*j+gamma+1] + 1
        elif numR-gamma <= symbol_bit*j and symbol_bit*j < numR + gamma:
          trainX[0,j,0:numR-symbol_bit*j+gamma] = -2*y[0, symbol_bit*j-gamma:numR] + 1
          trainX[0,j,numR-symbol_bit*j+gamma:] = 0

    return trainX

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

