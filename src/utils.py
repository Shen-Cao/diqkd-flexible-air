import numpy as np
import sys
sys.path.insert(0, '../submodules/py_aff3ct/build/lib')
import py_aff3ct
from py_aff3ct.module.py_module import Py_Module
import threading, time

DTYPE_FLOAT = np.float32

def wilson_ci(x, n, z):
    """Wilson score interval for binomial proportion."""
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / np.sqrt(denom)
    return center - half, center + half


def fmt_pct(x, prec=4):
    return f"{x*100:.{prec}f}%"

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
    # NOTE: dtype is explicit so that S == 0 yields an *empty integer* array
    # (a default-constructed empty np.array is float64 and cannot index).
    punct_set = np.array([reverse_bits(i, n_bits) for i in indices], dtype=np.int64)
    
    # Last S indices in sorted order are the punctured positions
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
    # NOTE: dtype is explicit so that S == 0 yields an *empty integer* array.
    punct_set = np.array([reverse_bits(i, n_bits) for i in indices], dtype=np.int64)
    
    # Last S indices in sorted order are the shortened positions
    return punct_set

def generate_shortened_from_file(file_path):
    """
    Read shortened positions from a file produced by polar-fbgen-rust.

    The on-disk format (written by cli.rs::write_shortened_file) is:

        # Shortened positions (count=S)
        idx_0
        idx_1
        ...
        idx_{S-1}

    Lines starting with '#' are treated as comments and skipped. Blank lines
    are ignored. The remaining lines must each contain a single integer index
    in [0, N).

    Parameters
    ----------
    file_path : str
        Path to the shortened positions file.

    Returns
    -------
    np.ndarray
        1-D int64 array of shortened position indices, same shape and dtype
        convention as ``generate_shortened_BGL`` so the two are drop-in
        interchangeable from the caller's perspective.
    """
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            positions.append(int(line))
    return np.array(positions, dtype=np.int64)


def generate_punctured_from_file(file_path):
    """
    Read punctured positions from a file produced by polar-fbgen-rust.

    The on-disk format mirrors ``generate_shortened_from_file``:

        # Punctured positions (count=P)
        idx_0
        idx_1
        ...
        idx_{P-1}

    (Note: as of polar-fbgen-rust's current output writer, the header line in
    punctured files may erroneously read ``# Shortened positions (count=P)``.
    This loader ignores header text entirely and only trusts the integer body
    + the caller-supplied filename, so the typo is harmless here.)

    Lines starting with '#' are treated as comments and skipped. Blank lines
    are ignored. The remaining lines must each contain a single integer index
    in [0, N).

    Parameters
    ----------
    file_path : str
        Path to the punctured positions file.

    Returns
    -------
    np.ndarray
        1-D int64 array of punctured position indices, same shape and dtype
        convention as ``generate_punctured_BGL`` so the two are drop-in
        interchangeable from the caller's perspective. Returns an empty
        1-D int64 array if the file contains no indices (P=0 case).
    """
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            positions.append(int(line))
    return np.array(positions, dtype=np.int64)

def reverse_bits(n, n_bits):
    """Reverse the bits of an integer with given bit length."""
    result = 0
    for i in range(n_bits):
        if n & (1 << i):
            result |= (1 << (n_bits - 1 - i))
    return result

def generate_frozen_bits_from_file(file_dir, num_frozen_bits, sp_positions, verbose = True):
    """
    Generate frozen bit positions from a reliability file.
    
    Parameters
    ----------
    file_dir: str
        Path to the reliability file (.pc format).
    num_frozen_bits: int
        Number of effective frozen bits to select from the reliability order,
        excluding ``sp_positions``.
    sp_positions: np.ndarray
        Array of "off-limits" positions (shortened ∪ punctured) that are
        already frozen by construction. These are excluded from the
        ``num_frozen_bits`` count and are appended to the final frozen set.
        For shortened-only codes, pass the shortened indices directly; for
        joint punctured+shortened codes, pass ``np.concatenate([punct, short])``.
        
    Returns
    -------
    np.ndarray
        Boolean array indicating frozen bit positions (length N).
    """
    
    N, _, new_frozen_positions = generate_frozen_position_from_file(file_dir, num_frozen_bits, sp_positions, verbose = verbose)

    frozen_positions = np.concatenate([sp_positions, new_frozen_positions])
    
    # Create boolean array
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_positions] = True
    
    return frozen_bits


def generate_frozen_position_from_file(file_dir, num_frozen_bits, sp_positions, verbose = True):
    """
    Generate effective frozen bit indices from a reliability file.

    Picks the ``num_frozen_bits`` least-reliable channels that are NOT in
    ``sp_positions``. By BGL construction, shortened/punctured positions are
    already incapable/overcapable bits and therefore implicitly frozen — they
    must be excluded from the "effective" frozen count to avoid double-counting.
    
    Parameters
    ----------
    file_dir: str
        Path to the reliability file (.pc format).
    num_frozen_bits: int
        Number of effective frozen bits to select (excludes sp_positions).
    sp_positions: np.ndarray
        Off-limits positions (shortened ∪ punctured) — these are skipped when
        walking the reliability order. Pass the union for joint S+P codes.
        
    Returns
    -------
    (int, np.ndarray, np.ndarray)
        Triple ``(N, sp_positions, effective_frozen_positions)``:
        - N: mother code length (inferred from reliability file size)
        - sp_positions: same array as input (for caller convenience)
        - effective_frozen_positions: indices of the ``num_frozen_bits``
          least-reliable channels not already in ``sp_positions``
    """
    # Read the 4th line of the file as reliability_sorted_idx
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        # The 4th line contains the reliability information
        reliability_line = lines[3].strip()
        # Convert to np.int64 array
        reliability_sorted_idx = np.array([int(x) for x in reliability_line.split()], dtype=np.int64)
    
    # Convert sp_positions to a Python set for O(1) membership tests.
    # The naive `idx not in numpy_array` is O(N) per check, making the loop
    # below O(N^2) — prohibitive at large N (e.g. N=2^20 takes minutes).
    sp_set = set(int(x) for x in np.asarray(sp_positions).ravel())

    # Walk reliability_sorted_idx from least reliable to most reliable
    # (reverse of the .pc file order, which is most-reliable first), skipping
    # any position already in sp_set. Stop once we have num_frozen_bits of them.
    available_indices = []
    for idx in reversed(reliability_sorted_idx):
        if int(idx) not in sp_set:
            available_indices.append(idx)
            if len(available_indices) == num_frozen_bits:
                break
    
    if verbose:
        print("Totally {} bits appended to frozen bits set".format(len(available_indices)))

    return len(reliability_sorted_idx), sp_positions, np.array(available_indices, dtype=np.int64)


def generate_chsh_flip_mask(alice_basis, bob_basis, exp_type, alice_result,
                            bob_result, complementary=False):
    """
    Build a per-subgroup optimal Bob-flip mask for Bell-test rounds.

    For each ``(alice_basis, bob_basis)`` subgroup within the Bell-test rounds
    (``exp_type == 0``), computes the observed correlation ``E`` and decides
    whether flipping Bob's outcome in that subgroup reduces the post-AIR-flip
    BER.

    Two conventions are supported via ``complementary``:

    - ``complementary=False`` (default, the convention used by the AIR
      reconciliation programs in ``src/eat-sep/``):
      The downstream pipeline applies the AIR flip (``a → 1-a``) on Alice,
      so the polar-code BER equals ``P(a_raw == b_flipped)`` (agreement rate).
      To minimise this, flip iff ``E > 0`` (agreement > disagreement).

    - ``complementary=True`` (no-AIR-flip convention):
      No AIR flip is applied; BER equals ``P(a != b')`` (disagreement rate).
      To minimise this, flip iff ``E < 0``.

    Both conventions yield the same optimal per-subgroup BER of
    ``min(agree, disagree) / n`` but the flip masks are complementary.

    Key-gen rounds (``exp_type == 1``) are never flipped — they are already
    strongly anti-correlated.

    Parameters
    ----------
    alice_basis, bob_basis : np.ndarray
        1-D integer arrays of basis labels.
    exp_type : np.ndarray
        1-D integer array; 0 = Bell-test round, 1 = key-gen round.
    alice_result, bob_result : np.ndarray
        1-D binary outcome arrays (after SNSPD correction on Bob's side).
    complementary : bool, optional
        If True, use the ``E < 0`` threshold (no-AIR-flip convention).
        If False, use the ``E > 0`` threshold (AIR-flip convention, default).

    Returns
    -------
    np.ndarray
        1-D boolean array; True at positions where Bob's outcome should be
        flipped.
    """
    flip_mask = np.zeros(len(alice_result), dtype=bool)
    chsh_mask = (exp_type == 0)
    if not chsh_mask.any():
        return flip_mask

    threshold = (lambda E: E < 0) if complementary else (lambda E: E > 0)

    for ab in np.unique(alice_basis[chsh_mask]):
        for bb in np.unique(bob_basis[chsh_mask]):
            mask = chsh_mask & (alice_basis == ab) & (bob_basis == bb)
            if not mask.any():
                continue
            a = alice_result[mask]
            b = bob_result[mask]
            n = int(mask.sum())
            agree = int(((a == b)).sum())
            E = (agree - (n - agree)) / n
            if threshold(E):
                flip_mask[mask] = True

    return flip_mask


def generate_chsh_flip_mask_from_basis(alice_basis, bob_basis, flip_subgroups):
    """
    Build a Bob-flip mask for Bell-test rounds from a predetermined set of
    subgroups, without inspecting the outcomes.

    ``flip_subgroups`` is an iterable of ``(alice_basis, bob_basis)`` pairs
    in which Bob flips his outcome — e.g. the agreement-dominated (E > 0)
    subgroups determined from calibration data. Unlike
    :func:`generate_chsh_flip_mask`, the experimental outcomes are not
    consulted, so no outcome information is used to steer the flip decision
    on the experimental data.

    Parameters
    ----------
    alice_basis, bob_basis : np.ndarray
        1-D integer arrays of basis labels.
    flip_subgroups : iterable of (int, int)
        ``(alice_basis, bob_basis)`` pairs whose rounds are flipped.

    Returns
    -------
    np.ndarray
        1-D boolean array; True at positions where Bob's outcome should be
        flipped.
    """
    flip_mask = np.zeros(len(alice_basis), dtype=bool)
    for ab, bb in flip_subgroups:
        flip_mask |= (alice_basis == ab) & (bob_basis == bb)
    return flip_mask


def _load_pe_from_file(file_dir):
    """
    Load per-channel ML decoding error probabilities from a ``*_pe.txt`` file.

    The on-disk format is one entry per line as ``<index>\\t<probability>``
    (e.g. produced by the degrading-upgrading quantization of Tal & Vardy).
    Lines starting with '#' and blank lines are skipped. The returned array
    is indexed directly by channel index, i.e. ``pe[i] = P_e(i)``.

    Parameters
    ----------
    file_dir : str
        Path to the ``*_pe.txt`` reliability file.

    Returns
    -------
    np.ndarray
        1-D ``float64`` array of length ``N`` where ``arr[i]`` is the ML
        decoding error probability of the ``i``-th polarized channel.
    """
    values = []
    with open(file_dir, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            values.append(float(parts[1]))
    return np.array(values, dtype=np.float64)


def union_failure_bound_from_file(file_dir, frozen_vector, sp_positions,
                                  num_frozen_bits):
    """
    Calculate the union upper bound of the frame decoding failure probability.

    The union bound on the frame error probability of a polar code is
    [Tal & Vardy 2013; Zhou et al. 2022, Eq. (5)]:

    .. math::  \\varepsilon \\leq \\sum_{i \\in \\mathcal{I}} P_e(i)

    where :math:`\\mathcal{I}` is the information set — positions that are
    **not frozen** (neither effective-frozen nor shortened/punctured) — and
    :math:`P_e(i)` is the ML decoding error probability of the ``i``-th
    polarized channel.

    Parameters
    ----------
    file_dir : str
        Path to the channel reliability file (``*_pe.txt`` format, as
        produced by the degrading-upgrading polar-code construction).
    frozen_vector : np.ndarray
        Boolean array of length ``N`` marking frozen positions. Should
        include **both** sp_positions (shortened+punctured) and effective-
        frozen positions, i.e. be consistent with the output of
        :func:`generate_frozen_bits_from_file`.
    sp_positions : np.ndarray
        1-D array of off-limits position indices (shortened ∪ punctured).
        These positions are preset by the protocol (shortened=0, punctured=
        erased), so they are not counted as leaked syndrome bits.
    num_frozen_bits : int
        Number of **effective** (non-sp) frozen bits. Used for consistency
        validation against ``frozen_vector`` and ``sp_positions``.

    Returns
    -------
    float
        Union upper bound :math:`\\sum_{i \\in \\mathcal{I}} P_e(i)` over
        the information set.

    Raises
    ------
    AssertionError
        If ``num_frozen_bits`` does not match the effective frozen count
        derived from ``frozen_vector`` minus ``sp_positions``, or if
        ``frozen_vector`` length disagrees with the ``*_pe.txt`` file.
    """
    pe = _load_pe_from_file(file_dir)
    N = len(pe)

    frozen_vector = np.asarray(frozen_vector, dtype=bool)
    sp_positions = np.asarray(sp_positions, dtype=np.int64)

    assert frozen_vector.shape[0] == N, (
        "frozen_vector length ({}) != N from pe file ({})".format(
            frozen_vector.shape[0], N)
    )

    sp_mask = np.zeros(N, dtype=bool)
    sp_mask[sp_positions] = True

    effective_frozen_count = int((frozen_vector & ~sp_mask).sum())
    assert effective_frozen_count == num_frozen_bits, (
        "num_frozen_bits mismatch: frozen_vector has {} effective frozen bits "
        "(frozen_vector.sum()={}, sp={}), but num_frozen_bits={}.".format(
            effective_frozen_count, int(frozen_vector.sum()),
            int(sp_mask.sum()), num_frozen_bits)
    )

    # frozen_vector from generate_frozen_bits_from_file already includes
    # sp_positions, so ~frozen_vector would suffice. We AND with ~sp_mask
    # for robustness against callers who pass a frozen vector that excludes
    # sp_positions.
    info_mask = ~frozen_vector & ~sp_mask

    return float(pe[info_mask].sum())


def num_frozen_bits_from_union_bound_from_file(file_dir, sp_positions,
                                               target_failure_rate):
    """
    Find the minimal number of effective frozen bits to achieve a target
    frame failure probability under the union bound.

    Given a target :math:`\\varepsilon_{\\mathrm{target}}`, finds the smallest
    number of effective (non-sp) frozen bits :math:`K_{\\min}` such that the
    union bound on the failure probability of the remaining information set
    satisfies:

    .. math::  \\sum_{i \\in \\mathcal{I}} P_e(i) \\leq \\varepsilon_{\\mathrm{target}}

    where :math:`|\\mathcal{I}| = L - K_{\\min}` and :math:`L` is the number
    of non-sp (transmitted+info-eligible) positions.

    The optimal frozen-set selection (minimizing :math:`K` for a given
    :math:`\\varepsilon`) is to freeze the :math:`K` positions with the
    **highest** :math:`P_e` (least reliable channels), leaving the
    :math:`L-K` positions with the **lowest** :math:`P_e` as the information
    set.

    Parameters
    ----------
    file_dir : str
        Path to the channel reliability file (``*_pe.txt`` format). For
        shortened/punctured polar codes, prefer the files in
        ``conf/polar/sp-bsc/`` which are computed with
        shortening/puncturing-aware channel modeling.
    sp_positions : np.ndarray
        1-D array of off-limits position indices (shortened ∪ punctured).
    target_failure_rate : float
        Upper bound on the frame failure probability to achieve
        (:math:`\\varepsilon_{\\mathrm{target}}`).

    Returns
    -------
    int
        Minimal number of effective (non-sp) frozen bits required.
        ``0`` if even with no frozen bits the union bound is below target;
        ``L`` (all non-sp positions frozen) if the target is too small.
    """
    pe = _load_pe_from_file(file_dir)
    N = len(pe)

    sp_positions = np.asarray(sp_positions, dtype=np.int64)
    sp_mask = np.zeros(N, dtype=bool)
    sp_mask[sp_positions] = True

    pe_ns = pe[~sp_mask]
    L = int(pe_ns.shape[0])

    pe_sorted = np.sort(pe_ns)
    cumsum = np.cumsum(pe_sorted)

    # side='right' counts cumsum entries that are <= target, which is the
    # largest K_info satisfying the union bound sum_{j<K_info} pe_sorted[j] <= target.
    K_info = int(np.searchsorted(cumsum, target_failure_rate, side='right'))

    return L - K_info


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


class NumpyArrayHttpReceiver:
    '''
    A class to receive numpy arrays over HTTP protocol.
    
    This class sets up an HTTP server that can receive numpy arrays sent by clients.
    The arrays are serialized and sent as JSON data with shape and dtype information.
    
    Parameters
    ----------
    host : str
        The host address to bind the server to
    port : int
        The port number to listen on
        
    Attributes
    ----------
    host : str
        The host address the server is bound to
    port : int
        The port number the server is listening on
    received_array : numpy.ndarray
        The most recently received numpy array
    server_running : bool
        Flag indicating if the server is currently running
    '''
    
    def __init__(self, host='localhost', port=8080):
        import json
        import http.server
        import socketserver
        import threading
        import socket
        from io import BytesIO
        import base64
        
        self.host = host
        self.port = port
        self.received_array = None
        self.server_running = False
        self.server = None
        self.server_thread = None
        
        # Store required modules for use in methods
        self.json = json
        self.http_server = http.server
        self.socketserver = socketserver
        self.threading = threading
        self.base64 = base64
        self.socket = socket
        
    def _create_handler(self):
        '''Create a custom request handler class'''
        json = self.json
        base64 = self.base64
        receiver_instance = self
        from io import BytesIO
        
        class ArrayRequestHandler(self.http_server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    # Parse the JSON data
                    data = json.loads(post_data.decode('utf-8'))
                    
                    # Extract array information
                    shape = tuple(data['shape'])
                    dtype = data['dtype']
                    array_data = base64.b64decode(data['data'])
                    
                    # Reconstruct the numpy array
                    array_bytes = BytesIO(array_data)
                    reconstructed_array = np.load(array_bytes)
                    reconstructed_array = reconstructed_array.reshape(shape)
                    
                    # Store the received array
                    receiver_instance.received_array = reconstructed_array
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"status": "success", "message": "Array received successfully"})
                    self.wfile.write(response.encode('utf-8'))
                    
                except Exception as e:
                    # Send error response
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"status": "error", "message": str(e)})
                    self.wfile.write(response.encode('utf-8'))
                    
            def log_message(self, format, *args):
                # Suppress default logging
                pass
                
        return ArrayRequestHandler
        
    def start_server(self):
        '''Start the HTTP server in a separate thread'''
        if self.server_running:
            print("Server is already running")
            return
            
        handler = self._create_handler()
        self.server = self.socketserver.TCPServer((self.host, self.port), handler)
        # Allow socket to be reused
        self.server.socket.setsockopt(self.socket.SOL_SOCKET, self.socket.SO_REUSEADDR, 1)
        self.server_thread = self.threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.server_running = True
        print(f"Server started at http://{self.host}:{self.port}")
        
    def stop_server(self):
        '''Stop the HTTP server'''
        if not self.server_running:
            print("Server is not running")
            return
            
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.server_running = False
        print("Server stopped")
        
    def get_received_array(self):
        '''Get the most recently received numpy array'''
        return self.received_array
        
    def receive_one_array(self, timeout=30):
        '''
        Start the server, receive exactly one array, stop the server and return that array.
        
        Parameters
        ----------
        timeout : int, optional
            Maximum time in seconds to wait for an array (default is 30 seconds)
            
        Returns
        -------
        numpy.ndarray or None
            The received numpy array, or None if timeout occurred
        '''
        import time
        
        # Reset any previously received array
        self.received_array = None
        
        # Start the server
        self.start_server()
        
        # Wait for an array to be received
        start_time = time.time()
        while self.received_array is None and (time.time() - start_time) < timeout:
            time.sleep(0.1)  # Check every 100ms
            
        # Stop the server
        self.stop_server()
        
        # Return the received array (or None if timeout)
        return self.received_array


class NumpyArrayHttpSender:
    '''
    A class to send numpy arrays over HTTP protocol.
    
    This class can send numpy arrays to an HTTP server. The arrays are serialized
    and sent as JSON data with shape and dtype information.
    
    Parameters
    ----------
    host : str
        The host address of the server to send to
    port : int
        The port number of the server to send to
        
    Attributes
    ----------
    host : str
        The host address of the server
    port : int
        The port number of the server
    '''
    
    def __init__(self, host='localhost', port=8080):
        import json
        import urllib.request
        from io import BytesIO
        import base64
        
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        
        # Store required modules for use in methods
        self.json = json
        self.urllib_request = urllib.request
        self.base64 = base64
        
    def send_array(self, array):
        '''
        Send a numpy array to the HTTP server
        
        Parameters
        ----------
        array : numpy.ndarray
            The numpy array to send
            
        Returns
        -------
        dict
            Response from the server
        '''
        from io import BytesIO
        
        # Convert array to bytes
        array_buffer = BytesIO()
        np.save(array_buffer, array)
        array_bytes = array_buffer.getvalue()
        
        # Create JSON payload
        payload = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'data': self.base64.b64encode(array_bytes).decode('utf-8')
        }
        
        json_data = self.json.dumps(payload).encode('utf-8')
        
        # Create HTTP request
        req = self.urllib_request.Request(
            self.url,
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'Content-Length': len(json_data)
            }
        )
        
        try:
            # Send the request and get response
            with self.urllib_request.urlopen(req) as response:
                response_data = response.read().decode('utf-8')
                return self.json.loads(response_data)
        except Exception as e:
            return {"status": "error", "message": str(e)}
