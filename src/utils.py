import numpy as np
import sys
sys.path.insert(0, '../submodules/py_aff3ct/build/lib')
import py_aff3ct
from py_aff3ct.module.py_module import Py_Module
import threading, time

DTYPE_FLOAT = np.float32

class seq_execution(threading.Thread):
    '''
    A thread for executing an AFF3CT sequence.

    Parameters
    ----------
    seq: The AFF3CT sequence to execute.
    '''

    def __init__(self, seq):
        # super().__init__(self)
        super(seq_execution, self).__init__()
        self.seq = seq
        self.starting_time = time.time()

    def run(self):
        self.starting_time = time.time()
        self.seq.exec()

class extractor(Py_Module):
    '''
    Tasks
    ----
    extract: extracts [idx_start : idx_stop] of the input data

    Parameters
    ----
    N_in: dimension of input
    idx_start: start index of extracting
    idx_stop: stop index of extracting
    dtype: dtype of input and output

    Sockets
    ----
    input: input vector, should be of shape (1, N_in)
    output: output vector of shape (1, idx_stop - idx_start)
    '''

    def extract(self, x_in, x_out):
        x_out[:, :] = x_in[:, self.idx_start:self.idx_stop]
        return 0

    def __init__(self, N_in, idx_start, idx_stop, dtype = np.int32):
        Py_Module.__init__(self)
        self.N_in = N_in
        self.idx_start = idx_start
        self.idx_stop = idx_stop
        self.N_out = self.idx_stop - self.idx_start
        self.dtype = dtype
        assert self.N_out > 0, 'idx_start should be smaller than idx_stop'
        self.name = "py_extractor"
        task_extract = self.create_task("extract")
        self.create_socket_in(task_extract, "input", N_in, self.dtype)
        self.create_socket_out(task_extract, "output", self.N_out, self.dtype)
        self.create_codelet(task_extract, lambda m,l,f: m.extract(l[0], l[1]))
    pass

# Quick check of the extractor
# in_np = np.random.randint(0, 12, (100,), dtype=np.int32).reshape((1, -1))
# extor = extractor(100, 10, 60)
# extor["extract :: input"].bind(in_np)
# extor.tasks[0].debug = True
# extor.tasks[0].exec()

class fillor(Py_Module):
    '''
    Tasks
    ----
    fillin: overwrite position [idx_start : idx_stop] of the input vector with another shorter input vector

    Parameters
    ----
    N_out: dimension of the output vector and the original input vector
    idx_start: start index of filling
    idx_stop: stop index of filling
    dtype: datatype of all vectors

    Sockets
    ----
    input_rx: the original vector to be modified, of shape (1, N_out)
    input_fill: the vector to be filled into the [idx_start, idx_stop] positions of input_rx vector, of shape (1, idx_stop - idx_start)
    output: the output vector, of shape (1, N_out)
    '''

    def fillin(self, x_fill, x_rx, x_out):
        x_out[:, :] = x_rx[:, :]
        x_out[:, self.idx_start:self.idx_stop] = x_fill[:, :]
        return 0

    def __init__(self, N_out, idx_start, idx_stop, dtype = np.int32):
        Py_Module.__init__(self)
        self.idx_start = idx_start
        self.idx_stop = idx_stop
        self.N_in = self.idx_stop - self.idx_start
        self.dtype = dtype
        assert self.N_in > 0, 'idx_start should be smaller than idx_stop'
        self.N_out = N_out
        self.name = "py_fillor"
        task_fillin = self.create_task("fillin")
        self.create_socket_in(task_fillin, "input_fill", self.N_in, self.dtype)
        self.create_socket_in(task_fillin, "input_rx", self.N_out, self.dtype)
        self.create_socket_out(task_fillin, "output", self.N_out, self.dtype)
        self.create_codelet(task_fillin, lambda m,l,f: m.fillin(l[0], l[1], l[2]))
    pass

# Quick check of the fillor
# in_np = np.random.randint(0, 12, (100,), dtype=np.int32).reshape((1, -1))
# in_fill = np.random.randint(100, 112, (50,), dtype=np.int32).reshape((1, -1))
# flor = fillor(100, 10, 60)
# flor["fillin :: input_fill"].bind(in_fill)
# flor["fillin :: input_rx"].bind(in_np)
# flor.tasks[0].debug = True
# flor.tasks[0].exec()

class splitter(Py_Module):
    '''
    Split the input vector into two parts.

    Tasks
    ----
    split: splits input vector at specified index

    Parameters
    ----
    N_in: dimension of input vector
    idx_stop: the index where the input vector will be split
    dtype: datatype of all vectors

    Sockets
    ----
    input: the input vector, of shape (1, N_in)
    output_1: the first output vector, of shape (1, idx_stop)
    output_2: the second output vector, of shape (1, N_in - idx_stop)
    '''

    def split(self, x_in, x_out_1, x_out_2):
        x_out_1[:, :] = x_in[:, :self.idx_stop_1]
        x_out_2[:, :] = x_in[:, self.idx_stop_1:]
        return 0

    def __init__(self, N_in, idx_stop, dtype = np.int32):
        Py_Module.__init__(self)
        self.N_in = N_in
        self.idx_stop_1 = idx_stop
        self.N_out_1 = self.idx_stop_1
        self.N_out_2 = self.N_in - self.idx_stop_1
        self.dtype = dtype
        assert self.N_out_2 > 0, ' '
        self.name = "py_splitter"
        task_split = self.create_task("split")
        self.create_socket_in(task_split, "input", N_in, self.dtype)
        self.create_socket_out(task_split, "output_1", self.N_out_1, self.dtype)
        self.create_socket_out(task_split, "output_2", self.N_out_2, self.dtype)
        self.create_codelet(task_split, lambda m,l,f: m.split(l[0], l[1], l[2]))
    pass

class concater(Py_Module):
    '''
    Concatenate two input vectors.

    Tasks
    ----
    concatenate: joins two input vectors

    Parameters
    ----
    N_in_1: Dimension of the first input vector
    N_in_2: Dimension of the second input vector
    dtype: datatype of the input vectors

    Sockets
    ----
    input_1: first input vector, of shape (1, N_in_1)
    input_2: second input vector, of shape (1, N_in_2)
    output: concatenated output vector, of shape (1, N_in_1 + N_in_2)
    '''

    def concatenate(self, x_in_1, x_in_2, x_out):
        x_out[:, :self.idx_stop] = x_in_1[:, :]
        x_out[:, self.idx_stop:] = x_in_2[:, :]
        return 0

    def __init__(self, N_in_1, N_in_2, dtype = np.int32):
        Py_Module.__init__(self)
        self.N_in_1 = N_in_1
        self.N_in_2 = N_in_2
        self.idx_stop = self.N_in_1
        self.N_out = self.N_in_1 + self.N_in_2
        self.dtype = dtype
        self.name = "py_concater"
        task_concat = self.create_task("concatenate")
        self.create_socket_in(task_concat, "input_1", N_in_1, self.dtype)
        self.create_socket_in(task_concat, "input_2", self.N_in_2, self.dtype)
        self.create_socket_out(task_concat, "output", self.N_out, self.dtype)
        self.create_codelet(task_concat, lambda m,l,f: m.concatenate(l[0], l[1], l[2]))
    pass

class XOR_er(Py_Module):
    '''
    Perform element-wise XOR operation.

    Tasks
    ----
    XOR: computes XOR of two binary vectors

    Parameters
    ----
    N: length of input vectors

    Sockets
    ----
    input_1: first binary input vector, of shape (1, N)
    input_2: second binary input vector, of shape (1, N)
    output: XOR result vector, of shape (1, N)
    '''

    def add_noise(self, x_in_1, x_in_2, x_out):
        
        np.logical_xor(x_in_1, x_in_2, out = x_out)
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_XOR"
        task_xor = self.create_task("XOR")
        socket_input_bits_1 = self.create_socket_in (task_xor, "input_1", N, np.int32)
        socket_input_bits_2 = self.create_socket_in (task_xor, "input_2", N, np.int32)
        socket_output_bits = self.create_socket_out(task_xor, "output", N, np.int32)
        self.create_codelet(task_xor, lambda m,l,f: m.add_noise(l[0], l[1], l[2]))
        
    pass

# Generate a random binary vector of length N with L ones on random positions
def generate_random_vector(N, L, dtype = np.int32):
    '''
    Generate a random binary vector with specified number of ones.

    Parameters
    ----
    N: length of vector
    L: number of ones in vector
    dtype: datatype of vector

    Returns
    ----
    np.ndarray: generated vector
    '''
    # Create a vector of zeros with int32 dtype
    vector = np.zeros(N, dtype = dtype)
    
    # Randomly select L unique positions
    positions = np.random.choice(N, size=L, replace=False)
    
    # Set those positions to 1
    vector[positions] = 1
    
    return vector

class mask_generator(Py_Module):
    '''
    Generate random binary masks.

    Tasks
    ----
    generate: creates random binary masks

    Parameters
    ----
    N: length of mask vector
    L: number of ones in mask
    dtype: datatype of mask

    Sockets
    ----
    output: generated mask vector, of shape (1, N)
    '''

    def generate(self, x_out):
        for l in range(x_out.shape[0]):
            mask = generate_random_vector(self.N, self.L, self.dtype)
            x_out[l, :] = mask
        return 0

    def __init__(self, N, L, dtype = np.int32):
        Py_Module.__init__(self)
        self.name = 'py_maskgenerator'
        task_generate = self.create_task('generate')
        self.N = N
        self.L = L
        self.dtype = dtype
        socket_output = self.create_socket_out(task_generate, "output", self.N, dtype)
        self.create_codelet(task_generate, lambda m,l,f: m.generate(l[0]))

class mask_fillor(Py_Module):
    '''
    Fill values into positions specified by a mask.

    Tasks
    ----
    fill: inserts values at mask positions

    Parameters
    ----
    N_in: length of input vector
    N_out: length of output vector
    dtype: datatype of vectors

    Sockets
    ----
    input: values to fill, of shape (1, N_in)
    mask: binary mask vector, of shape (1, N_out)
    output: filled vector, of shape (1, N_out)
    '''

    def fill(self, x_in, mask, x_out):
        # Initialize output with zeros
        # x_out = np.zeros_like(mask)
        
        # Get indices where mask is 1
        # indices = np.where(mask == 1)[0]
        indices = np.where(mask == 1)
        
        # Fill output positions where mask is 1 with input values
        x_out[indices] = x_in
        
        return 0

    def __init__(self, N_in, N_out, dtype=np.int32):
        Py_Module.__init__(self)
        self.name = "py_maskfill"
        task_fill = self.create_task("fill")
        self.N_in = N_in
        self.N_out = N_out
        self.dtype = dtype
        socket_input = self.create_socket_in(task_fill, "input", N_in, dtype)
        socket_mask = self.create_socket_in(task_fill, "mask", N_out, np.int32)
        socket_output = self.create_socket_out(task_fill, "output", N_out, dtype)
        self.create_codelet(task_fill, lambda m,l,f: m.fill(l[0], l[1], l[2]))
    pass

class mask_selector(Py_Module):
    '''
    Select values using a binary mask.

    Tasks
    ----
    select: extracts values at mask positions

    Parameters
    ----
    N_in: length of input vector
    N_out: length of output vector
    dtype: datatype of vectors

    Sockets
    ----
    input: input vector, of shape (1, N_in)
    mask: binary mask vector, of shape (1, N_in)
    output: selected values, of shape (1, N_out)
    '''

    def select(self, x_in, mask, x_out):
        # Initialize output with zeros
        # x_out = np.zeros_like(mask)
        
        # Get indices where mask is 1
        # indices = np.where(mask == 1)[0]
        indices = np.where(mask == 1)
        
        # Fill output positions where mask is 1 with input values
        x_out[:] = (x_in[indices])[:]
        
        return 0

    def __init__(self, N_in, N_out, dtype=np.int32):
        Py_Module.__init__(self)
        self.name = "py_maskselect"
        task_select = self.create_task("select")
        self.N_in = N_in
        self.N_out = N_out
        self.dtype = dtype
        socket_input = self.create_socket_in(task_select, "input", N_in, dtype)
        socket_mask = self.create_socket_in(task_select, "mask", N_in, np.int32)
        socket_output = self.create_socket_out(task_select, "output", N_out, dtype)
        self.create_codelet(task_select, lambda m,l,f: m.select(l[0], l[1], l[2]))
    pass


# We omit the modulate process for Binary Symmetric Channel
# Instead, we implement a customized BSC model
class BSC_digital_p(Py_Module):
    '''
    Simulate a Binary Symmetric Channel.

    Tasks
    ----
    digital_bit_flip: flips bits with probability p_c

    Parameters
    ----
    p_c: crossover probability
    N: length of input vector

    Sockets
    ----
    input_bits: input binary vector, of shape (1, N)
    output_bits: output binary vector, of shape (1, N)
    '''

    def add_noise(self, x_in, x_out):
        # Generate the error position 
        error_bit_string = np.random.binomial(n=1, p=self.p_c, size=x_out.shape)
        # Flip the bits in x_o based on the error_bit_string
        # x_out[:] = np.mod(x_in[:] + error_bit_string[:], 2).astype(DTYPE_FLOAT)
        x_out[:] = np.mod(x_in[:] + error_bit_string[:], 2).astype(np.int32)
        return 0

    def __init__(self, p_c, N):
        Py_Module.__init__(self)
        self.p_c = p_c
        self.name = "py_BSC_digital_p"
        task_bit_flip = self.create_task("digital_bit_flip")
        socket_input_bits = self.create_socket_in (task_bit_flip, "input_bits", N, np.int32)
        # socket_output_bits = self.create_socket_out(task_bit_flip, "output_bits", N, DTYPE_FLOAT)
        socket_output_bits = self.create_socket_out(task_bit_flip, "output_bits", N, np.int32)
        self.create_codelet(task_bit_flip, lambda m,l,f: m.add_noise(l[0], l[1]))
        
    pass

# Quickly test the module
# BSCer = BSC_digital_p(0.01, 512)
# BSCer.tasks[0].debug = True
# len_ = 512
# input_bits = np.random.randint(0, 2, len_, np.int32).reshape((1, -1))
# # print(np.shape(BSCer["digital_bit_flip::input_bits"][:]))
# BSCer["digital_bit_flip::input_bits"].bind(input_bits)
# BSCer("digital_bit_flip").exec()

class BSC_BPSK_vec(Py_Module):
    '''
    Apply BSC noise to BPSK-modulated signal.

    Tasks
    ----
    BPSK_bit_flip_vec: flips BPSK symbols based on flip vector

    Parameters
    ----
    N: length of input vector

    Sockets
    ----
    input_bits: BPSK-modulated input, of shape (1, N)
    flip_bits: binary flip vector, of shape (1, N)
    output_bits: noisy output, of shape (1, N)
    '''

    def add_noise(self, x_in, x_flip, x_out):
        # Generate the error position 
        sig_multiplication = x_flip.astype(DTYPE_FLOAT) * (-2) + 1
        # Flip the bits in x_o based on the error_bit_string
        # x_out[:] = np.mod(x_in[:] + error_bit_string[:], 2).astype(DTYPE_FLOAT)
        x_out = x_in * sig_multiplication
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_BSC_BPSK_vec"
        task_bit_flip = self.create_task("BPSK_bit_flip_vec")
        socket_input_bits = self.create_socket_in (task_bit_flip, "input_bits", N, DTYPE_FLOAT)
        socket_flip_bits = self.create_socket_in (task_bit_flip, "flip_bits", N, np.int32)
        socket_output_bits = self.create_socket_out(task_bit_flip, "output_bits", N, DTYPE_FLOAT)
        self.create_codelet(task_bit_flip, lambda m,l,f: m.add_noise(l[0], l[1], l[2]))
        
    pass


# We need a module to store the data for analyze 
class vec2file_np(Py_Module):
    '''
    Save vectors to file (not implemented).

    Tasks
    ----
    vec_2_file: saves vectors to file

    Parameters
    ----
    N: vector dimension
    filename: output filename
    in_dtype: input datatype
    save_dtype: storage datatype
    buffer_size: buffer size
    compression: compression method

    Sockets
    ----
    input: input vector to save
    '''

    def to_file(self, x_in):
        return 0

    def __init__(self, N, filename, in_dtype = np.int32, save_dtype = np.bool, buffer_size = 1000, compression='gzip'):
        Py_Module.__init__(self)
        self.name = "py_vector_2_file"
        self.dim = N
        self.filename = filename
        self.in_dtype = in_dtype
        self.save_dtype = save_dtype
        self.buffer_size = buffer_size
        self.compression = compression

        assert compression in ['gzip', 'lzf']
        
        self.buffer = np.ndarray((self.buffer_size, self.dim), self.save_dtype)
        self.total_vectors = 0
        self.file_handle = None
        self.dataset = None

        task_vec2file = self.create_task("vec_2_file")
        raise NotImplementedError("This module is not implemented yet.")
    pass

def reverse_bits(n, lgN):
    m = 0
    i = 0
    while i < lgN:
        m = (m << 1) + (n & 1)
        n >>= 1
        i += 1
    return m

class bit_reversal(Py_Module):
    '''
    Perform bit-reversal permutation.

    Tasks
    ----
    BR_permutation: applies bit-reversal permutation

    Parameters
    ----
    N: vector length (must be power of 2)
    dtype: datatype of vector

    Sockets
    ----
    input: input vector, of shape (1, N)
    output: permuted vector, of shape (1, N)
    '''

    def bit_reversal_permutation(self, input_bits, output_bits):
        # Some of the idx will not be reversed
        output_bits[:] = input_bits[:]
        for idx in range(self.N):
            reversed_idx = reverse_bits(idx, self.lgN)
            if reversed_idx > idx :
                output_bits[:, reversed_idx] = input_bits[:, idx]
                output_bits[:, idx] = input_bits[:, reversed_idx]
        
        # output_bits[:] = input_bits[:]
        return 0

    def __init__(self, N, dtype = np.int32):
        Py_Module.__init__(self) 
        self.N = N
        assert self.N & self.N-1 == 0, "Input dimension must be power of 2."
        self.lgN = int(np.log2(self.N))
        self.dtype = dtype
        task_BRP = self.create_task("BR_permutation")
        socket_input = self.create_socket_in(task_BRP, "input", self.N, self.dtype)
        socket_output = self.create_socket_out(task_BRP, "output", self.N, self.dtype)
        self.create_codelet(task_BRP, lambda m, l, f: m.bit_reversal_permutation(l[0], l[1]))

        return 

    pass


class signal_waiter2(Py_Module):
    '''
    Wait for second input before outputting first input.

    Tasks
    ----
    wait: outputs first input when second input arrives

    Parameters
    ----
    in1_dim: dimension of first input
    in2_dim: dimension of second input
    in1_dtype: datatype of first input
    in2_dtype: datatype of second input

    Sockets
    ----
    input_1: first input vector
    input_2: second input vector
    output: output vector (same as input_1)
    '''

    def do_nothing(self, input_1, input_2, output):
        "output input1 when receive input2"
        output[:, :] = input_1[:, :]
        return 0
    
    def __init__(self, in1_dim, in2_dim, in1_dtype = np.int32, in2_dtype = np.int32):
        Py_Module.__init__(self)

        self.name = "sig_waiter2"
        self.in1_dim = in1_dim
        self.in2_dim = in2_dim
        self.in1_dtype = in1_dtype
        self.in2_dtype = in2_dtype

        tsk = self.create_task("wait")
        skt_input1 = self.create_socket_in (tsk, "input_1", self.in1_dim, self.in1_dtype)
        skt_input2 = self.create_socket_in (tsk, "input_2", self.in2_dim, self.in2_dtype)
        skt_output = self.create_socket_out (tsk, "output", self.in1_dim, self.in1_dtype)
        self.create_codelet(tsk, lambda m,l,f: m.do_nothing(l[0], l[1], l[2]))

class linear_amplifier(Py_Module):
    '''
    Amplify input signal by a control factor.

    Tasks
    ----
    amplify: multiplies input by control factor

    Parameters
    ----
    N: length of input vector
    dtype: datatype of input and output
    control_dtype: datatype of control factor

    Sockets
    ----
    input: input vector, of shape (1, N)
    control: control factor, of shape (1, 1)
    output: amplified output vector, of shape (1, N)
    '''

    def amplify(self, input_1, control, output):
        "output input_1 * control"
        output[:] = input_1[:] * np.asarray(control, dtype=self.dtype).reshape((-1))[0]
        return 0
    
    def __init__(self, N, dtype = DTYPE_FLOAT, control_dtype = DTYPE_FLOAT):
        Py_Module.__init__(self)

        self.name = "py_linear_amplifier"
        self.N = N
        self.dtype = dtype
        self.control_dtype = control_dtype

        task_amplify = self.create_task("amplify")
        skt_input_1 = self.create_socket_in(task_amplify, "input", self.N, self.dtype)
        skt_input_2 = self.create_socket_in(task_amplify, "control", 1, self.control_dtype)
        skt_output = self.create_socket_out(task_amplify, "output", self.N, self.dtype)
        self.create_codelet(task_amplify, lambda m,l,f: m.amplify(l[0], l[1], l[2]))

class control_identity(Py_Module):
    '''
    Output input only when control is non-zero.

    Tasks
    ----
    control_identity: outputs input when control is non-zero

    Parameters
    ----
    N: length of input vector
    dtype: datatype of input and output
    control_dtype: datatype of control signal

    Sockets
    ----
    input: input vector, of shape (1, N)
    control: control signal, of shape (1, 1)
    output: output vector, of shape (1, N)
    '''

    def control_identity(self, input_1, control, output):
        "output input_1 iff control is not 0"
        output[:] = input_1[:] * np.asarray(np.asarray(control, dtype=np.bool), dtype=self.dtype).reshape((-1))[0]
        return 0
    
    def __init__(self, N, dtype = DTYPE_FLOAT, control_dtype = DTYPE_FLOAT):
        Py_Module.__init__(self)

        self.name = "py_linear_amplifier"
        self.N = N
        self.dtype = dtype
        self.control_dtype = control_dtype

        task_identity = self.create_task("control_identity")
        skt_input_1 = self.create_socket_in(task_identity, "input", self.N, self.dtype)
        skt_input_2 = self.create_socket_in(task_identity, "control", 1, self.control_dtype)
        skt_output = self.create_socket_out(task_identity, "output", self.N, self.dtype)
        self.create_codelet(task_identity, lambda m,l,f: m.control_identity(l[0], l[1], l[2]))

class control_flusher(Py_Module):
    '''
    Output input when control is zero, otherwise output ones.

    Tasks
    ----
    control_identity: outputs input or ones based on control

    Parameters
    ----
    N: length of input vector
    dtype: datatype of input and output
    control_dtype: datatype of control signal

    Sockets
    ----
    input: input vector, of shape (1, N)
    control: control signal, of shape (1, 1)
    output: output vector, of shape (1, N)
    '''

    def control_identity(self, input_1, control, output):
        "work as a identity iff control is 0, otherwise output a one vector"
        output[:] = (input_1 * np.asarray(np.asarray(control, dtype=np.bool), dtype=self.dtype).reshape((-1))[0] + np.ones_like(input_1) * np.asarray(np.logical_not(np.asarray(control, dtype=np.bool)), dtype=self.dtype).reshape((-1))[0])[:]
        return 0
    
    def __init__(self, N, dtype = DTYPE_FLOAT, control_dtype = DTYPE_FLOAT):
        Py_Module.__init__(self)

        self.name = "py_linear_amplifier"
        self.N = N
        self.dtype = dtype
        self.control_dtype = control_dtype

        task_identity = self.create_task("control_identity")
        skt_input_1 = self.create_socket_in(task_identity, "input", self.N, self.dtype)
        skt_input_2 = self.create_socket_in(task_identity, "control", 1, self.control_dtype)
        skt_output = self.create_socket_out(task_identity, "output", self.N, self.dtype)
        self.create_codelet(task_identity, lambda m,l,f: m.control_identity(l[0], l[1], l[2]))

def generate_punctured_BGL(N, S):
    """
    Generate S punctured positions for a polar code of length N (power of two).
    
    Parameters:
    N (int): Block length (must be power of two)
    S (int): Number of punctured positions

    
    Returns:
    list: Indices of punctured positions
    """
    n_bits = int(np.log2(N))
    indices = np.array(range(0, S))
    
    # Compute bit-reversed values
    punct_set = np.array([reverse_bits(i, n_bits) for i in indices])
    
    # Last S indices in sorted order are the shortened positions
    return punct_set

def generate_shortened_BGL(N, S):
    """
    Generate S shortened positions for a polar code of length N (power of two).
    
    Parameters:
    N (int): Block length (must be power of two)
    S (int): Number of shortened positions

    
    Returns:
    list: Indices of shortened positions
    """
    n_bits = int(np.log2(N))
    indices = np.array(range(N - S, N))
    
    # Compute bit-reversed values
    punct_set = np.array([reverse_bits(i, n_bits) for i in indices])
    
    # Last S indices in sorted order are the shortened positions
    return punct_set

def reverse_bits(n, n_bits):
    """Reverse the bits of an integer with given bit length."""
    result = 0
    for i in range(n_bits):
        if n & (1 << i):
            result |= (1 << (n_bits - 1 - i))
    return result

def generate_frozen_bits_from_file(file_dir, num_frozen_bits, shortened_positions, verbose = True):
    """
    Generate frozen bit positions from a reliability file.
    
    Parameters
    ----------
    file_dir: str
        Path to the reliability file
    num_frozen_bits: int
        Number of frozen bits to select
    shortened_positions: np.ndarray
        Array of shortened positions (these should also be frozen)
        
    Returns
    -------
    np.ndarray
        Boolean array indicating frozen bit positions
    """
    
    N, _, new_frozen_positions = generate_frozen_position_from_file(file_dir, num_frozen_bits, shortened_positions, verbose = verbose)

    frozen_positions = np.concatenate([shortened_positions, new_frozen_positions])
    
    # Create boolean array
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_positions] = True
    
    return frozen_bits


def generate_frozen_position_from_file(file_dir, num_frozen_bits, shortened_positions, verbose = True):
    """
    Generate frozen bit positions from a reliability file.
    
    Parameters
    ----------
    file_dir: str
        Path to the reliability file
    num_frozen_bits: int
        Number of frozen bits to select
    shortened_positions: np.ndarray
        Array of shortened positions (these should also be frozen)
        
    Returns
    -------
    np.ndarray
        Shortened positions and new frozen positions
    """
    # Read the 4th line of the file as reliability_sorted_idx
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        # The 4th line contains the reliability information
        reliability_line = lines[3].strip()
        # Convert to np.int64 array
        reliability_sorted_idx = np.array([int(x) for x in reliability_line.split()], dtype=np.int64)
    
    # Get the last num_frozen_bits elements that are not in shortened_positions
    # We need to filter out shortened positions from the reliability indices
    available_indices = []
    for idx in reversed(reliability_sorted_idx):  # Process in reverse order (least reliable first)
        if idx not in shortened_positions:
            available_indices.append(idx)
            if len(available_indices) == num_frozen_bits:
                break
    
    if verbose:
        print("Totally {} bits appended to frozen bits set".format(len(available_indices)))

    return len(reliability_sorted_idx), shortened_positions, np.array(available_indices, dtype=np.int64)

class shortener(Py_Module):
    '''
    Add large LLR values to specified positions.

    Tasks
    ----
    shorten: adds large LLR values to specified positions

    Parameters
    ----
    N: length of input vector
    add_LLR: LLR value to add

    Sockets
    ----
    input: input vector, of shape (1, N)
    output: modified output vector, of shape (1, N)
    '''

    def shorten(self, x_in, x_out):
        x_out[:] = (x_in + ((self.shorten_vec * x_in) * self.add_LLR))[:]
        return 0

    def __init__(self, N, add_LLR = 16.):
        Py_Module.__init__(self)
        self.name = "py_shortener"
        task_shorten = self.create_task("shorten")
        self.shorten_vec = np.zeros(shape = (1, N), dtype = DTYPE_FLOAT)
        self.add_LLR = add_LLR
        socket_input_bits = self.create_socket_in (task_shorten, "input", N, DTYPE_FLOAT)
        socket_output_bits = self.create_socket_out(task_shorten, "output", N, DTYPE_FLOAT)
        self.create_codelet(task_shorten, lambda m,l,f: m.shorten(l[0], l[1]))
    
    def set_shorten_bits(self, shorten_vec):

        # We need some check here...
        self.shorten_vec = shorten_vec
        return 
        
    pass

class puncturer(Py_Module):
    '''
    Set specified positions to zero.

    Tasks
    ----
    puncture: zeros out specified positions

    Parameters
    ----
    N: length of input vector

    Sockets
    ----
    input: input vector, of shape (1, N)
    output: modified output vector, of shape (1, N)
    '''

    def puncture(self, x_in, x_out):
        x_out[:] = (x_in - (self.punct_vec * x_in))[:]
        return 0

    def __init__(self, N):
        Py_Module.__init__(self)
        self.name = "py_puncturer"
        task_puncture = self.create_task("puncture")
        self.punct_vec = np.zeros(shape = (1, N), dtype = DTYPE_FLOAT)
        socket_input_bits = self.create_socket_in (task_puncture, "input", N, DTYPE_FLOAT)
        socket_output_bits = self.create_socket_out(task_puncture, "output", N, DTYPE_FLOAT)
        self.create_codelet(task_puncture, lambda m,l,f: m.puncture(l[0], l[1]))
    
    def set_punct_bits(self, punct_vec):

        # We need some check here...
        self.punct_vec = punct_vec
        return 
        
    pass

class llr_calculater_bsc(Py_Module):
    '''
    Calculate LLR values for BSC channel.

    Tasks
    ----
    llr_bsc: computes LLR values

    Parameters
    ----
    p: crossover probability
    N: length of input vector
    dtype: datatype of input

    Sockets
    ----
    input: input binary vector, of shape (1, N)
    output: LLR values, of shape (1, N)
    '''

    def llr(self, x_in, x_out):
        x_out[:] = (1 - 2 * x_in.astype(DTYPE_FLOAT)[:]) * self.p_coef
        return 0

    def __init__(self, p, N, dtype = np.int32):
        Py_Module.__init__(self)
        self.name = "py_llr_bsc"
        self.dtype = dtype
        self.p_coef = np.log((1-p)/p)
        task_llr = self.create_task("llr_bsc")
        socket_input_bits = self.create_socket_in (task_llr, "input", N, self.dtype)
        socket_output_bits = self.create_socket_out(task_llr, "output", N, DTYPE_FLOAT)
        self.create_codelet(task_llr, lambda m,l,f: m.llr(l[0], l[1]))
        
    pass

class bit_reverser(Py_Module):
    '''
    Apply bit-reversal permutation to input.

    Tasks
    ----
    reverse_bits: applies bit-reversal permutation

    Parameters
    ----
    N: length of input vector
    dtype: datatype of input and output

    Sockets
    ----
    input: input vector, of shape (1, N)
    output: permuted output vector, of shape (1, N)
    '''

    def bit_reverse_transform(self, x_in, x_out):
        x_out[:, :] = x_in[:, self.reversed_indices]
        return 0

    def __init__(self, N, dtype = np.int32):
        Py_Module.__init__(self)
        self.name = "py_bit_reverser"
        self.N = N
        self.dtype = dtype
        self.reversed_indices = np.int64([reverse_bits(idx, n) for idx in range(self.N)])
        task_br = self.create_task("reverse_bits")
        socket_input_bits = self.create_socket_in (task_br, "input", N, self.dtype)
        socket_output_bits = self.create_socket_out(task_br, "output", N, self.dtype)
        self.create_codelet(task_br, lambda m,l,f: m.bit_reverse_transform(l[0], l[1]))
        
    pass
