import numpy as np

def LDPC_decoder(c_llr, H, iter_num, early_stop = False):
  #   c_llr = channel llr values
  #   H = parity check matrix
  #   iter_num = total iteration number

  size_H = H.shape
  n = size_H[-1]
  m = size_H[0]

  # initiliaze the messages
  VN_to_CN = H*c_llr

  for iter in range(1,iter_num+1):
      CN_to_VN = np.zeros(size_H)

      # Calculate VN to CN messages
      for i in range(m):

          # Get current row vector of H
          curr_row = H[i,:]
          # Find where this equals to non-zero
          ind = np.squeeze(np.argwhere(curr_row))
          # get messages coming to CN
          mes = VN_to_CN[i,ind]

          for j in range(len(ind)):

              # Delete the corresponding CN_to_VN meesage where we send info
              mes_upd = np.copy(mes)
              mes_upd = np.delete(mes_upd, j)

              # Find current CN to VN message
              CN_to_VN[i, ind[j]] = 2*np.arctanh(np.clip(np.prod(np.tanh(mes_upd/2)), -0.9999999999, 0.9999999999))

      # If early stop is true
      if early_stop:
        # Estimate the messages
        llr = np.sum(CN_to_VN, 0) + c_llr
        m_est = llr < 0
        if np.sum(np.mod(m_est @ H.T, 2)) == 0:
            #print('Process is stopped at iteration: ', iter)
            break

      VN_to_CN = np.zeros(size_H)
      # Calculate VN to CN messages
      for j in range(n):
          # Get current column vector of H
          curr_col = H[:,j]
          # Find where this equals to non-zero
          ind = np.squeeze(np.argwhere(curr_col))
          # get messages coming to CN
          mes = CN_to_VN[ind, j]

          for i in range(len(ind)):
              # Delete the corresponding CN_to_VN meesage where we send info
              mes_upd = np.copy(mes)
              mes_upd = np.delete(mes_upd, i)

              # Find current VN to CN message
              VN_to_CN[ind[i],j] = np.sum(mes_upd)+c_llr[0,j]

  # Estimate the messages
  llr = np.sum(CN_to_VN, 0) + c_llr
  m_est = llr < 0

  return m_est, llr
