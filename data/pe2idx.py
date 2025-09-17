import numpy as np

# # # # # # # # # # # # # # # # 
# Generate channel indices sorted by channel reliability
# # # # # # # # # # # # # # # # 

pe_file_txt = './conf/polar/bsc/n17_bsc_ber0.036_mu40_pe_cpp.txt'
out_best_channels_file = './conf/polar/bsc/n17_bsc_ber0.036_mu40_cpp.pc'

pe_v = np.loadtxt(pe_file_txt, delimiter="\t", usecols=1, dtype=np.float64)
sorted_idx = np.argsort(pe_v)
np.savetxt(out_best_channels_file, sorted_idx, fmt="%d", delimiter='', newline=' ')