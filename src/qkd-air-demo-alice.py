#%% Import necessary libraries
import sys
import os
sys.path.insert(0, '../submodules/py_aff3ct/build/lib')
import numpy as np
import time
from datetime import datetime

import py_aff3ct

assumed_q_ber = 0.0375

# Log the program
log_dir = '../log/demo'
os.makedirs(log_dir, exist_ok=True)
start_time = datetime.now()
log_file = os.path.join(log_dir, f'qkdir_demo_alice_assumed_{assumed_q_ber}_{start_time.strftime("%H%M%S")}.txt')

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)
            
    def flush(self):
        self.terminal.flush()
        with open(self.log_file, 'a') as f:
            f.flush()

sys.stdout = Logger(log_file)

print("*"*32)
print("Reconciliation of real sifted KEY data")
print("This log file collects the operation records in information reconciliation and error verification of Alice and Bob, respectively.")
print("The operations performed on the two parts are labeled by \"Alice :\" and \"Bob :\" respectively")
print("Remark: The statistical QBER is only for calculating the efficiency in this demo, which is manually set")
print("        Alice and Bob will only read their own KEY")
print("\n")

with open('../data/Alice_key.bin', 'rb') as f: 
    data = np.fromfile(f, dtype = np.uint8)
    pass

alice_sifted_key = np.unpackbits(data)

alice_sifted_key = alice_sifted_key.astype(np.int32)

# assert alice_sifted_key.shape[0] == bob_sifted_key.shape[0], "sifted key should of equal length."

len_sifted_key = alice_sifted_key.shape[0]

#%% Configure Superparameters
# # # # # # # # # # # # # # # #
num_samples = 1
# Polar codes parameters
# Assumed bit error rate
q_ber =  assumed_q_ber

# Polar code Block size
# The polar code block size can be determined automatically
n = np.ceil(np.log2(len_sifted_key)).astype(np.int64)
block_encoded_bits = 2 ** n
# The length of the list of SCL Decoder will affect the performance of the SCL Decoder ---- and running time of codes
scl_len = 16

# CRC check
crc_size = 32

# Quantum bit error rate from experimental data
# q_ber_stat = np.logical_xor(alice_sifted_key, bob_sifted_key).astype(np.int32).sum() / len_sifted_key
q_ber_stat = 0.03597984449309278

print("*"*32)
# print("Quantum Bit Error Rate (QBER) from dataset: {} / {} = {}".format(
#     np.logical_xor(alice_sifted_key, bob_sifted_key).astype(np.int32).sum(), len_sifted_key, q_ber_stat))
print("Real Quantum Bit Error Rate (QBER) from dataset: {}".format(q_ber_stat))
print("*"*32)
print("Configured QBER: {}".format(q_ber))
if np.abs(q_ber - q_ber_stat) > 0.001:
    print("*"*32)
    print("Deviation between assumed QBER and statistical QBER: {}".format(np.abs(q_ber - q_ber_stat)))
    print("*"*32)

#%% Calculate Efficiency
# # # # # # # # # # # # # # # #
shannon_H2 = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x)
ideal_frozen_bits = len_sifted_key * shannon_H2(q_ber_stat)
assumed_ideal_frozen_bits = len_sifted_key * shannon_H2(q_ber)
crc_efficiency_gain = crc_size / ideal_frozen_bits
assumed_crc_efficiency_gain = crc_size / assumed_ideal_frozen_bits

print('Statistical Qubit Error Rate: {} -> H_2 = {}'.format(q_ber_stat, shannon_H2(q_ber_stat)))
print('----> Estimated Quantum Channel Leaked Bits (L * H_2(QBER)): {}'.format(ideal_frozen_bits))
print('----> Real Reconciliation Efficiency = Number of Leaked Bits / {}'.format(ideal_frozen_bits))
print('\n')

# For reconciling experiment data, finer efficiency stages should be configured to obtain better efficiency performance
configured_efficiency_stage = np.arange(1.080, 1.1001, 0.002, dtype = np.float64)
# CRC tag will also contribute to total efficiency
efficiency_stage = configured_efficiency_stage - assumed_crc_efficiency_gain
# efficiency_stage = configured_efficiency_stage - crc_efficiency_gain

effective_frozen_bits_stage = np.asarray(efficiency_stage * assumed_ideal_frozen_bits, dtype = np.int64)
effective_efficiency_stage = effective_frozen_bits_stage.astype(np.float64) / assumed_ideal_frozen_bits
# effective_frozen_bits_stage = np.asarray(efficiency_stage * ideal_frozen_bits, dtype = np.int64)
# effective_efficiency_stage = effective_frozen_bits_stage.astype(np.float64) / ideal_frozen_bits

# The number of shortened bits will be the same in all stages
shortened_bits = block_encoded_bits - len_sifted_key
info_bits_stage = block_encoded_bits - effective_frozen_bits_stage - shortened_bits

print("*"*32)
print("Preparation Phase: The following information and parameters are pre-shared")
print('-' * 32)
print('Length of Sifted KEY: {} \nLength of Polar Codes: {}'.format(len_sifted_key, block_encoded_bits))
print('----> Number of Shortened Bits: {}'.format(shortened_bits))
# print('Statistical Qubit Error Rate: {} -> H_2 = {}'.format(q_ber_stat, shannon_H2(q_ber_stat)))
# print('----> Estimated Quantum Channel Leaked Bits (L * H_2(QBER)): {}'.format(len_sifted_key * shannon_H2(q_ber_stat)))
print('Configured Qubit Error Rate: {} -> H_2 = {}'.format(q_ber, shannon_H2(q_ber)))
print('----> Estimated Least Polar Code Frozen Bits (L * H_2(Assumed_QBER)): {}'.format((len_sifted_key) * shannon_H2(q_ber)))
print('----> Estimated Reconciliation Efficiency = Number of Leaked Bits / {}'.format(assumed_ideal_frozen_bits))
print("-"*32)
print("Configured Efficiency stages: {}".format(configured_efficiency_stage))
# print("Configured Efficiency stages: {}".format(efficiency_stage))
print("-"*32)
print("Effecctive Efficiency Stages (without CRC tag): {}".format(effective_efficiency_stage))
print("Assumed Number of bits leaked by CRC Tag: {}. Efficiency gain: {}".format(crc_size, assumed_crc_efficiency_gain))
# print("Number of bits leaked by CRC Tag: {}. Efficiency gain: {}".format(crc_size, crc_efficiency_gain))
print("----> Effective Efficiency Stages: {}".format(effective_efficiency_stage + assumed_crc_efficiency_gain))
# print("----> Effective Efficiency Stages: {}".format(effective_efficiency_stage + crc_efficiency_gain))
print('-' * 32)
print('Enc-Dec Effective Information Bits Stages: {}'.format(info_bits_stage))
print('Enc-Dec Effective Frozen Bits Stages:      {}'.format(effective_frozen_bits_stage))
print('-' * 32)
print('\n')

#%% Prepare Frozen Vector
# # # # # # # # # # # # # # # #
sys.path.insert(0, '../src/')
from utils import generate_shortened_BGL, generate_frozen_position_from_file
from copy import deepcopy

fb_file_dir = '../conf/polar/bsc/n{}_bsc_ber{}_mu40.pc'.format(int(np.log2(block_encoded_bits)), q_ber)

try:
    with open(fb_file_dir, 'r') as f:
        lines = f.readlines()
        # The 4th line contains the reliability information
        reliability_line = lines[3].strip()
        # Convert to np.int64 array
        reliability_sorted_idx = np.array([int(x) for x in reliability_line.split()], dtype=np.int64)
except Exception:
    # print("polar code .pc file not found")
    # print("try to load _cpp.pc file")
    fb_file_dir = '../conf/polar/bsc/n{}_bsc_ber{}_mu40_cpp.pc'.format(int(np.log2(block_encoded_bits)), q_ber)
    with open(fb_file_dir, 'r') as f:
        lines = f.readlines()
        # The 4th line contains the reliability information
        reliability_line = lines[3].strip()
        # Convert to np.int64 array
        reliability_sorted_idx = np.array([int(x) for x in reliability_line.split()], dtype=np.int64)
shortened_idx = generate_shortened_BGL(block_encoded_bits, shortened_bits)
shortened_vec = np.zeros(block_encoded_bits, dtype=bool)
shortened_vec[shortened_idx] = np.ones_like(shortened_vec[shortened_idx])

frozen_vec_stage = []
effective_frozen_vec_stage = []

for effective_frozen_bits_nums in effective_frozen_bits_stage: 
    _, __, effective_frozen_idx = generate_frozen_position_from_file(fb_file_dir, effective_frozen_bits_nums, shortened_idx, False)
    frozen_bits = np.zeros(block_encoded_bits, dtype=bool)
    effective_frozen_bits = np.zeros(block_encoded_bits, dtype=bool)
    frozen_bits[shortened_idx] = np.ones_like(frozen_bits[shortened_idx]).astype(np.bool)
    frozen_bits[effective_frozen_idx] = np.ones_like(frozen_bits[effective_frozen_idx]).astype(np.bool)
    effective_frozen_bits[effective_frozen_idx] = np.ones_like(effective_frozen_bits[effective_frozen_idx]).astype(np.bool)
    frozen_vec_stage.append(deepcopy(frozen_bits))
    effective_frozen_vec_stage.append(deepcopy(effective_frozen_bits))
    pass

print('\n')
print("Number of Total Frozen Bits of Each Stage: {}".format([int(a.astype(np.int32).sum()) for a in frozen_vec_stage]))
print("Number of Effective Frozen Bits of Each Stage: {}".format([int(a.astype(np.int32).sum()) for a in effective_frozen_vec_stage]))
print('\n')

#%% Prepare py_aff3ct Modules
# Polar Encoder and CRC-aided SCL Decoder
# # # # # # # # # # # # # # # #
from py_aff3ct.module.encoder import Encoder_polar
from py_aff3ct.module.decoder import Decoder_polar_SCL_naive_CA, Decoder_polar_SCL_naive
from py_aff3ct.module.crc import CRC, CRC_polynomial
crc_alice = CRC_polynomial(block_encoded_bits, '0x04C11DB7', crc_size)
enc_stage = []
enc_stage.append(Encoder_polar(info_bits_stage[0], block_encoded_bits, frozen_vec_stage[0]))
enc_stage[0].set_frozen_bits(frozen_vec_stage[0])

#%% Initialize al py_aff3ct Modules
# # # # # # # # # # # # # # # #
from utils import BSC_digital_p, extractor, signal_waiter2, mask_fillor, mask_selector
from py_aff3ct.module.monitor import Monitor_BFER_AR

# The padder will fill the sifted KEY bits into non-shortened positions
non_shortened_mask = np.logical_not(np.asarray(shortened_vec, dtype=np.bool)).astype(np.int32).reshape((1, -1))
padder_alice = mask_fillor(len_sifted_key, block_encoded_bits, dtype=np.int32)

# We also need another selector to extract the frozen bits in the encoded alice vector
effective_frozen_vec_mask_stage = [deepcopy(bl.astype(np.int32).reshape((1, -1))) for bl in effective_frozen_vec_stage]

# An extractor will be utilized to extract the crc value in output message of CRC "build" method
ext_crc_stage = []
ext_crc_stage.append(extractor(block_encoded_bits + crc_size, block_encoded_bits, block_encoded_bits + crc_size))

#%% Execute
# # # # # # # # # # # # # # # #
from utils import NumpyArrayHttpReceiver, NumpyArrayHttpSender
# Alice sends to Bob on port 8003 and receives from Bob on port 8002
sender_host = '10.64.86.101'
sender_port = 8003
receiver_host = '10.68.133.215'
receiver_port = 8002
array_sender = NumpyArrayHttpSender(host=sender_host, port=sender_port)
array_receiver = NumpyArrayHttpReceiver(host=receiver_host, port=receiver_port)
array_receiver.start_server()

fixed_alice_from_file = alice_sifted_key.reshape((1, -1)).astype(np.int32)
# Alice and Bob execute step by step
num_bits_transmitted = 0
print("# "*16)
print(f"({datetime.now() - start_time}) Reconciliation Phase")
print("The operations performed on the two parts are labeled by \"Alice :\" and \"Bob :\" respectively")
print("All communications are transmitted via classical public noiseless channel")
print("# "*16)
print("\n")

array_sender.send_array(np.ones(2))
print("Alice: Start the reconciliation phase")
print("\n")

# Alice pad the KEY
padded_alice_bits = np.zeros((1, block_encoded_bits), dtype = np.int32)
padder_alice["fill ::output"].bind(padded_alice_bits)
padder_alice["fill ::mask"].bind(non_shortened_mask)
padder_alice["fill ::input"].bind(fixed_alice_from_file)
print(f"({datetime.now() - start_time}) Alice: Pad sifted KEY")
padder_alice["fill"].exec()
print(f"Alice: Length: {len_sifted_key} to {block_encoded_bits}")
print(f"Alice: Padded sifted KEY: {padded_alice_bits}")
print("\n")

# Alice apply the polar encoder to padded sifted key
encoded_alice = np.zeros_like(padded_alice_bits)
enc_stage[0]["light_encode ::U_N"].bind(padded_alice_bits)
enc_stage[0]["light_encode ::X_N"].bind(encoded_alice)
enc_stage[0]["light_encode"].exec()
print(f"({datetime.now() - start_time}) Alice: Encode padded sifted KEY")
print(f"Alice: Encoded KEY: {encoded_alice}")
print("\n")

# Alice prepare the CRC tag then send it to Bob
encoded_alice_crc = np.zeros((1, block_encoded_bits + crc_size), dtype=np.int32)
crc_alice["build ::U_K1"].bind(encoded_alice)
crc_alice["build ::U_K2"].bind(encoded_alice_crc)
crc_alice["build"].exec()
alice_crc_tag = np.zeros((1, crc_size), dtype = np.int32)
ext_crc_stage[0]["extract ::input"].bind(encoded_alice_crc)
ext_crc_stage[0]["extract ::output"].bind(alice_crc_tag)
ext_crc_stage[0]["extract"].exec()
print(f"({datetime.now() - start_time}) Alice: Calculate CRC tag")
print(f"Alice: CRC tag of encoded sifted KEY: {alice_crc_tag}")
print("\n")

time.sleep(5)

# Alice send CRC tag to Bob
print("*"*8)
print(f"({datetime.now() - start_time}) TRANSMISSION: Alice to Bob")
print(f"{crc_size} bits of CRC tag")
num_bits_transmitted += crc_size
array_sender.send_array(alice_crc_tag)
time.sleep(10)
print(f"Total {num_bits_transmitted} bits has been made public")
print("*"*8)
print("\n")

failure_flag = True

# Appending Reconciliation
for stage in range(len(frozen_vec_stage)):
    print("-"*16)
    print(f"({datetime.now() - start_time}) Stage {stage+1} of appending reconciliation")
    print("-"*16)
    print("\n")

    # Alice append new frozen bits, then send it to Bob
    print(f"({datetime.now() - start_time}) Alice: Extract appended frozen bits")
    if stage == 0:
        num_appended_bits = effective_frozen_vec_stage[stage].astype(np.int32).sum()
        encoded_alice_frozen_bits = np.zeros((1, num_appended_bits), dtype = np.int32)
        indices = np.where(effective_frozen_vec_mask_stage[stage] == 1)
    else:
        num_appended_bits = effective_frozen_vec_stage[stage].astype(np.int32).sum() -\
                        effective_frozen_vec_stage[stage-1].astype(np.int32).sum()
        encoded_alice_frozen_bits = np.zeros((1, num_appended_bits), dtype = np.int32)
        indices = np.where(np.logical_xor(
            effective_frozen_vec_mask_stage[stage], 
            effective_frozen_vec_mask_stage[stage-1]).astype(np.int32) == 1)
        pass
    encoded_alice_frozen_bits[:] = (encoded_alice[indices])[:]
    # print("Alice: Extracted frozen bits: {}".format(encoded_alice_frozen_bits))
    # print("Alice: Number of extracted frozen bits: {}".format(effective_frozen_vec_stage[stage].astype(np.int32).sum()))
    print(f"Alice: Appended frozen bits: {encoded_alice_frozen_bits}")
    print(f"Alice: Number of appended bits: {num_appended_bits}")
    print("\n")

    print("*"*8)
    print(f"({datetime.now() - start_time}) TRANSMISSION: Alice to Bob")
    print(f"{num_appended_bits} bits of appended frozen bits")
    num_bits_transmitted += num_appended_bits
    array_sender.send_array(encoded_alice_frozen_bits)
    # time.sleep(10)
    print(f"Total {num_bits_transmitted} bits has been made public")
    print("*"*8)
    print("\n")
    
    print("*"*8)
    print("TRANSMISSION: Bob to Alice")
    print("Alice: Waiting for CRC check result from Bob...")
    http_start_time = time.time()
    while array_receiver.received_array is None and (time.time() - http_start_time) < 120:
        time.sleep(0.1)  # Check every 100ms
    if array_receiver.received_array is not None:
        dec_status = array_receiver.received_array.copy()
        array_receiver.received_array = None  # Reset for next array
    else:
        print("Alice: Timeout waiting for dec_status from Bob")
        dec_status = None
    # dec_status = array_receiver.receive_one_array(timeout = 120)
    print(dec_status)
    print("1 bits of CRC check flag")
    num_bits_transmitted += 1
    print(f"Total {num_bits_transmitted} bits has been made public")
    print("*"*8)
    time.sleep(5)

    if np.abs(dec_status).sum() == 0:
        failure_flag = False
        print(f"({datetime.now() - start_time}) Success of reconciliation announced by Bob.")
        print("Assumed Effective Efficiency: {}".format(num_bits_transmitted/assumed_ideal_frozen_bits))
        print("Real Efficiency: {}".format(num_bits_transmitted/ideal_frozen_bits))
        break
    else: 
        print(f"Reconciliation FAILURE announced at stage {stage + 1}")
        pass
    print("Total leaked bits: {}".format(num_bits_transmitted))
    print("Assumed Effective Efficiency: {}".format(num_bits_transmitted/assumed_ideal_frozen_bits))
    print("Real Efficiency: {}".format(num_bits_transmitted/ideal_frozen_bits))
    print("\n")
    pass

if failure_flag:
    print("-"*16)
    print(f"({datetime.now() - start_time}) Final stage completed. Reconciliation Failed")
    pass
# At the end of the file, stop the receiver
array_receiver.stop_server()