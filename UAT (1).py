#!/usr/bin/env python3
"""
Single-User OFDMA with 16QAM @ Rate=0.75, using LDPC + 16-bit CRC.
Part 1 of 3: Imports & Basic Helper Functions

"""

import os
import uhd
from uhd.types import TimeSpec, TXMetadata, RXMetadata, RXMetadataErrorCode
from uhd.usrp import MultiUSRP, StreamArgs
import numpy as np
import threading
import time
import sys
import math
import binascii
import ldpc  # from the ldpc library you provided
from ldpc.bplsd_decoder import BpLsdDecoder
import matplotlib.pyplot as plt


# ----------------------------
# Basic Bits/Bytes + CRC Utils
# ----------------------------

def bits_to_bytes(bits):
    """
    Convert bits (0/1 array) to bytes, big-endian style.
    bits[0] is the MSB of the first byte.
    bits must be multiple of 8 in length.
    """
    bits = np.array(bits, dtype=np.uint8)
    if len(bits) % 8 != 0:
        raise ValueError("bits_to_bytes: length must be multiple of 8.")
    out = bytearray(len(bits)//8)
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | (bits[i + j] & 1)
        out[i//8] = val
    return bytes(out)

def crc16_ccitt(data_bytes, poly=0x1021, init=0xFFFF):
    """
    Standard 16-bit CRC-CCITT, polynomial=0x1021, init=0xFFFF, no reflection.
    """
    crc = init
    for b in data_bytes:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def add_crc16_bits(info_bits):
    """
    1) Pad info_bits to multiple of 8,
    2) Convert to bytes,
    3) Compute 16-bit CRC,
    4) Append those 16 bits in big-endian style.
    """
    info_bits = np.array(info_bits, dtype=np.uint8)
    remainder = len(info_bits) % 8
    if remainder != 0:
        pad_len = 8 - remainder
        info_bits = np.concatenate([info_bits, np.zeros(pad_len, dtype=np.uint8)])
    data_bytes = bits_to_bytes(info_bits)
    crc_val = crc16_ccitt(data_bytes)
    crc_bits = []
    mask = 1 << 15
    for _ in range(16):
        crc_bits.append(1 if (crc_val & mask) else 0)
        mask >>= 1
    crc_bits = np.array(crc_bits, dtype=np.uint8)
    return np.concatenate([info_bits, crc_bits])

def check_crc16_bits(decoded_bits):
    """
    The last 16 bits are the CRC in big-endian.
    1) separate payload from those 16 bits,
    2) payload -> bytes,
    3) compute CRC => compare with appended bits.
    Return (error_count, crc_ok, payload_bits).
    """
    decoded_bits = np.array(decoded_bits, dtype=np.uint8)
    if len(decoded_bits) < 16:
        return (16, False, decoded_bits)
    payload_len = len(decoded_bits) - 16
    payload = decoded_bits[:payload_len]
    crc_appended = decoded_bits[payload_len:]
    # Rebuild integer
    rec_crc_val = 0
    for b in crc_appended:
        rec_crc_val = (rec_crc_val << 1) | b
    # Pad payload to multiple of 8, then bits->bytes
    remainder = len(payload) % 8
    if remainder != 0:
        pad_len = 8 - remainder
        payload = np.concatenate([payload, np.zeros(pad_len, dtype=np.uint8)])
    data_bytes = bits_to_bytes(payload)
    calc_crc = crc16_ccitt(data_bytes)
    # compare
    error_count = 0
    mask = 1 << 15
    for b in crc_appended:
        bit_val = 1 if (calc_crc & mask) else 0
        if b != bit_val:
            error_count += 1
        mask >>= 1
    crc_ok = (error_count == 0)
    return (error_count, crc_ok, payload)


# ----------------------------
# SNR / QAM Tools
# ----------------------------

def Q_function(x):
    return 0.5 * math.erfc(x / math.sqrt(2))

def approximate_raw_ber(snr_linear, modulation):
    """
    Approximate uncoded bit error rate for AWGN, given snr_linear and modulation.
    """
    mod_dict = {'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
    if modulation not in mod_dict:
        return 0.01
    b = mod_dict[modulation]
    if modulation in ['BPSK','QPSK']:
        return Q_function(math.sqrt(2*snr_linear))
    else:
        M = 2**b
        Ms = int(math.sqrt(M))
        return (4.0/b)*(1 - 1.0/Ms)*Q_function(math.sqrt((3*b/(M-1))*snr_linear))

# ----------------------------
# Subcarrier Partition / RU 
# ----------------------------

def compute_used_indices(fft_size):
    used = []
    for k in range(fft_size):
        f = k if k < fft_size//2 else k - fft_size
        if -117 <= f <= 116:
            used.append(k)
    return np.array(sorted(used))

'''
def partition_RUs(used, ru_size):
    ru_groups = []
    num_full = len(used) // ru_size
    for i in range(num_full):
        group = used[i*ru_size:(i+1)*ru_size]
        ru_groups.append(group)
    return ru_groups

def process_RU(ru_group):
    ru = {"indices": ru_group, "pilot_positions": [5, 20], "data_positions": []}
    for i in range(len(ru_group)):
        if i not in ru["pilot_positions"]:
            ru["data_positions"].append(i)
    return ru

def assign_RUs_to_UE(all_RUs, N_UE):
    ue_RUs = [[] for _ in range(N_UE)]
    for i, ru in enumerate(all_RUs):
        ue_RUs[i % N_UE].append(ru)
    return ue_RUs

'''

def circulant_permutation(z, shift):
    """Generate a z×z circulant permutation matrix with the given shift."""
    I = np.eye(z, dtype=np.uint8)
    return np.roll(I, shift % z, axis=1)

'''
def construct_qc_ldpc(base_matrix, z):
    """
    Build an (m*z) × (n*z) parity-check matrix from 'base_matrix' of shape (m,n),
    with each base_matrix[i,j] in [-1, 0..z-1].
    - If entry is -1 => zero block
    - Else => circulant shift block
    """
    m, n = base_matrix.shape
    H = np.zeros((m*z, n*z), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            val = base_matrix[i, j]
            if val >= 0:
                block = circulant_permutation(z, val)
                H[i*z:(i+1)*z, j*z:(j+1)*z] = block
    return H

'''


def build_generator_matrix(H, m, n, z):
    """
    For a systematic QC-LDPC code with dimension (m*z) x (n*z),
    the code rate is (n-m)/n. Then K=(n-m)*z is the # info bits.
    If the right side of H is the identity portion, we can build G=[I, A^T],
    with A = the left block of H. 
    """
    rows, cols = H.shape
    K = cols - rows  # (n-m)*z
    A = H[:, :K]     # the left block
    I_K = np.eye(K, dtype=np.uint8)
    G = np.hstack((I_K, A.T))
    return G

#######################################
# Modulation / Demodulation
#######################################
def modulate(bits, modulation):
    """
    Hard-coded support for BPSK, QPSK, 16QAM, 64QAM, 256QAM
    Return array of complex symbols.
    """
    bits = np.array(bits, dtype=np.uint8)
    mod_dict = {'BPSK':1, 'QPSK':2, '16QAM':4, '64QAM':6, '256QAM':8}
    if modulation not in mod_dict:
        raise ValueError("Unknown modulation: " + modulation)
    b = mod_dict[modulation]
    if len(bits) % b != 0:
        raise ValueError("Length of bits must be multiple of " + str(b))
    import math
    symbols = []
    if modulation == 'BPSK':
        return (1.0 - 2.0*bits).astype(np.float64)
    elif modulation == 'QPSK':
        const = 1/math.sqrt(2)
        for i in range(0, len(bits), 2):
            b0, b1 = bits[i], bits[i+1]
            # We'll do Gray-coded QPSK => 00 => (1+1j)
            # or we can do the mapping used previously. We'll keep it simple:
            re = (1.0 if b0==0 else -1.0)
            im = (1.0 if b1==0 else -1.0)
            symbols.append((re + 1j*im)*const)
        return np.array(symbols, dtype=np.complex64)
    else:
        M = 2**b
        Ms = int(math.sqrt(M))
        # Normalization factor 
        norm_factor = math.sqrt( (2*(Ms**2 -1))/3 )
        for i in range(0, len(bits), b):
            accum=0
            for bb in bits[i:i+b]:
                accum= (accum <<1)|bb
            # accum is in [0..M-1], interpret as row/col
            row= accum>> (b//2)
            col= accum & ((1<<(b//2))-1)
            # e.g. for 16QAM => row= accum>>2, col= accum & 3
            # map to Gray-coded coordinates => simplest approach => 
            # "f= col - (Ms-1)/2, g= row - (Ms-1)/2" 
            # We'll do a standard approach:
            # but let's do the "lowest bits => row"? It's consistent with typical 
            re = ( (accum % Ms) - (Ms-1)/2 )
            im = ( (accum //Ms) - (Ms-1)/2 )
            sym = (re + 1j* im)/ norm_factor
            symbols.append(sym)
        return np.array(symbols, dtype=np.complex64)

def demodulate(symbols, modulation):
    """
    Hard decision demod for BPSK, QPSK, 16QAM, etc.
    Return array of bits.
    """
    mod_dict = {'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
    if modulation not in mod_dict:
        raise ValueError("demodulate: unknown modulation " + modulation)
    b= mod_dict[modulation]
    symbols = np.array(symbols, dtype=np.complex64)
    import math
    
    if modulation=='BPSK':
        # threshold=0 => 0 => +1, 1 => -1
        return (symbols<0).astype(np.uint8)
    elif modulation=='QPSK':
        bits=[]
        c=1/math.sqrt(2)
        # Each symbol => 2 bits
        for s in symbols:
            re = s.real
            im = s.imag
            #  re>=0 => bit0=0 else 1
            #  im>=0 => bit1=0 else 1
            b0= 0 if re>=0 else 1
            b1= 0 if im>=0 else 1
            bits.extend([b0,b1])
        return np.array(bits, dtype=np.uint8)
    else:
        bpp= b
        M= 2**bpp
        Ms= int(math.sqrt(M))
        norm_factor= math.sqrt((2*(Ms**2-1))/3)
        # build a small array of constellation points
        const_points=[]
        for i in range(M):
            re= (i % Ms)-(Ms-1)/2
            im= (i //Ms)-(Ms-1)/2
            val= (re + 1j*im)/norm_factor
            const_points.append(val)
        const_points= np.array(const_points, dtype=np.complex64)
        bits_out=[]
        for s in symbols:
            # find nearest point
            dists= np.abs(const_points- s)
            idx= np.argmin(dists)
            # convert idx => bpp bits
            bb= format(idx, '0{}b'.format(bpp))
            for c in bb:
                bits_out.append(int(c))
        return np.array(bits_out, dtype=np.uint8)

#############################################################################
# 4) The main encode/decode: encode_qc3_11ax() / decode_qc3_11ax()
#############################################################################
def encode_qc3_11ax(info_bits, num_subcarriers, modulation, code_rate=0.5):
    """
    Build the final parity-check matrix for the chosen code_rate
    from the TOY_80211AX_MATRICES dictionary. Then systematically encode.
    Steps:
      1) look up the base matrix (12 x columns).
      2) pick L= floor( (num_subcarriers*b)/ n )
      3) build final (12L x nL) matrix
      4) G= [I | A^T], with A= left block => shape => (12L, (n-12)L)
      5) pad/trunc info_bits => length => K= (n-12)*L
      6) encode => modulate => return (final_info, codeword, mod_signal, H)
    """
    # pick the base matrix from dictionary
    base_mat = None
    for r,mat in TOY_80211AX_MATRICES.items():
        if abs(r - code_rate)<1e-3:
            base_mat= mat
            break
    if base_mat is None:
        raise ValueError(f"No toy base matrix found for code_rate={code_rate}.")

    #global SMALL_R12_BASE
    #base_mat = np.array(SMALL_R12_BASE, dtype=int)  # shape (4,8)
    #base_mat= np.array(base_mat, dtype=int)
    m, n= base_mat.shape  # e.g. (12,24) for rate=1/2
    
    # figure out T= num_subcarriers*b
    mod_dict= {'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
    if modulation not in mod_dict:
        raise ValueError("encode_qc3_11ax: unknown modulation.")
    b= mod_dict[modulation]
    T= num_subcarriers*b
    # lifting factor
    L= T// n
    if L<=0:
        raise ValueError("encode_qc3_11ax: not enough subcarriers for this code rate.")
    
    # final codeword length => n*L
    # info bits => (n- m)*L
    K_3= (n - m)* L
    if K_3<=0:
        raise ValueError("encode_qc3_11ax: no room for info bits, K_3=0 or negative.")
    
    # build H_3 => shape => (mL, nL)
    H_3= np.zeros((m*L, n*L), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            val= base_mat[i,j]
            if val<0:
                # zero block
                continue
            block_start_row= i*L
            block_start_col= j*L
            Iblock= np.eye(L, dtype=np.uint8)
            # SHIFT= val mod L
            block= np.roll(Iblock, val%L, axis=1)
            H_3[block_start_row:block_start_row+L, block_start_col:block_start_col+L]= block
    
    # build generator => G= [I_K| A^T], A= left block => shape => (mL, K_3)
    A= H_3[:, :K_3]
    I_K= np.eye(K_3, dtype=np.uint8)
    G_3= np.hstack([I_K, A.T])
    
    # pad or truncate user info_bits => length => K_3
    user_len= len(info_bits)
    if user_len> K_3:
        final_info_bits= info_bits[:K_3].copy()
    elif user_len< K_3:
        pad_len= K_3- user_len
        final_info_bits= np.concatenate([info_bits, np.zeros(pad_len,dtype=np.uint8)])
    else:
        final_info_bits= info_bits.copy()
    
    # FEC encode
    codeword_3= np.mod(final_info_bits @ G_3, 2).astype(np.uint8)
    
    # modulate => user must define modulate(...) 
    mod_signal_3= modulate(codeword_3, modulation)
    
    return final_info_bits, codeword_3, mod_signal_3, H_3



def decode_qc3_11ax(received_signal, H_qc, final_info_len, modulation, SNR_dB=None, aware=True):
    """
    LSD decode approach. 
    1) demod => bits_det
    2) if syndrome=0 => done, else LSD with error_rate= approximate from SNR if aware
    3) recover => first K bits => unpad => final_info_len
    """
    bits_det= demodulate(received_signal, modulation)
    nrows, ncols= H_qc.shape
    K= ncols- nrows
    
    # syndrome
    syn= np.mod(H_qc @ bits_det, 2)
    if np.all(syn==0):
        corrected= bits_det.copy()
    else:
        from ldpc.bplsd_decoder import BpLsdDecoder
        error_rate= 0.0
        if aware and (SNR_dB is not None):
            snr_lin= 10**(SNR_dB/10)
            error_rate= approximate_raw_ber(snr_lin, modulation)
        bp_osd= BpLsdDecoder(H_qc,
                             error_rate=error_rate,
                             bp_method='product_sum',
                             max_iter=10,
                             schedule='serial',
                             lsd_method='lsd_cs',
                             lsd_order=0)
        est= bp_osd.decode(syn)
        corrected= np.mod(bits_det + est, 2)
    rec_info= corrected[:K]
    if final_info_len< K:
        rec_info= rec_info[: final_info_len]
    return corrected, rec_info

#########################################
# Part 3: Single-User Main Function, TX/RX Threads, LS Channel Estimation
#########################################

def tx_continuous(device, tx_waveform, TX_FREQ, tx_sampling_rate, tx_channel, tx_gain, duration):
	"""
	Continuously transmit 'tx_waveform' for 'duration' seconds.
	End with an empty burst. 
	"""
	st_args = StreamArgs("fc32", "sc16")
	st_args.channels = tx_channel
	tx_streamer = device.get_tx_stream(st_args)
	start_time = time.time()
	total_sent = 0
	while time.time() - start_time < duration:
		device.send_waveform(tx_waveform, duration, TX_FREQ, tx_sampling_rate, tx_channel, tx_gain)
		total_sent += 1
	# End burst
	#md = TXMetadata()
	#md.start_of_burst = False
	#md.end_of_burst = True
	#tx_streamer.send(np.empty((1,0), dtype=np.complex64), md)
	print("TX: Continuous transmission complete. Total samples sent =", total_sent)

def rx_continuous(device, chunk_size, RX_FREQ, rx_sampling_rate, rx_channel, rx_gain, duration, out_folder):
    """
    Continuously receive samples in chunks of size chunk_size for 'duration' seconds
    using device.recv_num_samps(). Each chunk is saved to a separate .bin file in out_folder.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    chunk_idx = 0
    start_time = time.time()
    total_samples = 0
    while time.time() - start_time < duration + 0.5:
        samples = device.recv_num_samps(chunk_size, RX_FREQ, rx_sampling_rate, rx_channel, rx_gain)
        if samples is not None and samples.size > 0:
            filename = os.path.join(out_folder, f"rx_chunk_{chunk_idx:04d}.bin")
            with open(filename, "wb") as f:
                f.write(samples.tobytes())
            print("RX: Wrote chunk", chunk_idx, "with", samples.size, "samples.")
            chunk_idx += 1
            total_samples += samples.size
        else:
            print("RX: No samples received in this chunk.")
    print("RX: Continuous reception complete. Total samples received =", total_samples)

def ls_channel_estimation_ofdm(freq_sym, pilot_positions, pilot_value=(1+1j)):
    """
    Simple LS channel estimation for a single RU (freq_sym is a 1D array of complex subcarriers).
    We average the ratio of (received pilot / pilot_value) at the pilot positions.
    Returns a single complex scalar h_est or 1+0j if no pilot found.
    """
    pilot_vals = []
    for p in pilot_positions:
        if p < len(freq_sym):
            pilot_vals.append(freq_sym[p] / pilot_value)
    if pilot_vals:
        return np.mean(pilot_vals)
    else:
        return 1+0j


def generate_sync_freq_symbols(fft_size, k1, k2,u_sync=1):
    
    # Total control subcarrier count:
    total_control = k1 + k2
    sync_length = fft_size - total_control
    if sync_length <= 0:
        raise ValueError("FFT size too small for given control subcarrier allocation.")
    
    # Generate sync portion using a Zadoff–Chu sequence (length = sync_length)
    n_sync = np.arange(sync_length)
    zc_sync = np.exp(-1j * np.pi * u_sync * n_sync * (n_sync + 1) / sync_length).astype(np.complex64)
    
    return zc_sync


def plot_constellation(modulation_type, symbols):
    """
    Plots the constellation diagram for the given modulation type and complex symbols.
    
    Parameters:
    - modulation_type: str, e.g. 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'
    - symbols: numpy array of complex numbers representing the modulated symbols
    """
    modulation_type = modulation_type.upper()
    
    # Create plot
    plt.figure(figsize=(6, 6))
    plt.scatter(symbols.real, symbols.imag, color='blue', s=20, alpha=0.6)
    
    # Plot settings
    plt.title(f'{modulation_type} Constellation Diagram')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True, linestyle='--')
    plt.axis('equal')

    # Optional: Set plot limits based on modulation type
    limits = {
        'BPSK': 1.5,
        'QPSK': 1.5,
        '16QAM': 3,
        '64QAM': 8,
        '256QAM': 18
    }
    
    if modulation_type in limits:
        plt.xlim(-limits[modulation_type], limits[modulation_type])
        plt.ylim(-limits[modulation_type], limits[modulation_type])
    
    plt.show()


def cfo_estimation_from_cp(rx_signal, fft_size, cp_size, sampling_rate):
    # Step 1: Reshape the received signal to get individual OFDM blocks
    L1 = len(rx_signal)
    L0 = fft_size + cp_size  # Length of one OFDM block (FFT size + CP size)
    
    # Step 2: Trim the rx_signal to fit exact number of blocks
    L2 = int(L1 / L0)  # Number of blocks
    rx_signal = rx_signal[:L2 * L0]  # Truncate any remaining samples that don't fit a full block
    
    # Step 3: Reshape the signal to (L0, L2) where L0 is the block size, and L2 is the number of blocks
    rx_signal_reshaped = np.reshape(rx_signal, (L0, L2))
    
    # Step 4: For each column (each OFDM block), estimate the CFO
    cfo_estimates = []
    
    for i in range(L2):
        candidate_block = rx_signal_reshaped[:, i]
        
        # Apply the simple CFO estimation method based on cyclic prefix
        cp_received = candidate_block[:cp_size]
        tail = candidate_block[-cp_size:]
        
        product = np.sum(cp_received * np.conjugate(tail))
        phase_diff = np.angle(product)
        
        # Estimate the CFO using the simple method
        f_est = -phase_diff / (2 * np.pi) * (sampling_rate / fft_size)
        
        cfo_estimates.append(f_est)
    
    return cfo_estimates


def process_rx_chunk_practical(out_folder,FFT_SIZE, SIMULATE):

    decoding_time_start = time.time()

    A1=commonTxRxParameters(FFT_SIZE)
    print("NFFT (Just to confirm) =",FFT_SIZE)

    TT=A1[0]
    KK=A1[1]
    YY=A1[2]
    GG=A1[3]
    k1=A1[4]
    k2=A1[5]
    subcarrier_spacing=A1[6]
    tx_sampling_rate=A1[7]
    rx_sampling_rate=A1[8]
    TX_FREQ=A1[9]
    RX_FREQ=A1[10]
    SYNC_DELAY=A1[11]
    pilot_positions=A1[12]
    pilot_value=A1[13]
    usedSCmin1=A1[14]
    usedSCmax1=A1[15]
    pilotPosWithInRSCMinSCMax1=A1[16]
    input_crc_size=A1[17]
    max_used_subcarriers_excluding_pilot=A1[18]
    CP_LEN=A1[19]

    sync_freq_subcarriers=generate_sync_freq_symbols(FFT_SIZE, k1, k2,u_sync=1)
    fft_size=FFT_SIZE
    rx_sampling_rate= FFT_SIZE* subcarrier_spacing
    cp_len=CP_LEN
    sampling_rate=rx_sampling_rate
    #GG=min(YY2,GG)
    block_size=2*(FFT_SIZE + CP_LEN)

    """
    Offline processing of a single chunk file:
        1) Read .bin file (complex64 samples).
        2) Split into OFDM-like symbols (of length CP_LEN+FFT_SIZE).
        3) For each symbol, remove CP, FFT, then do simple LS channel est on pilot positions.
            Then extract data subcarriers => demod => LSD decode => check CRC.
    For demonstration, we handle one RU (size RU_SIZE) in the lower subcarriers (e.g. subcarriers [0..RU_SIZE-1]).
    The rest are zero.
    """

    #files = [f for f in os.listdir(out_folder) if os.path.isfile(f)]
    #files = [f for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]
    files = [os.path.join(out_folder, f) for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]

    print("All saved Rx files are:")

	#if not SIMULATE:

    if SIMULATE:
        print("This is for simulation")
    else:
        files=sorted(files)
        tL1=int(0.5*len(files))
        files=files[tL1-int(0.5*tL1):tL1+int(0.5*tL1)]
        print(files)

    success1=0
    tempPayload1=[]

    for filename in files:
        if success1==1:
            break
        rx_data = np.fromfile(filename, dtype=np.complex64)
        symbol_len = CP_LEN + FFT_SIZE
        num_symbols = len(rx_data) // symbol_len
        if num_symbols == 0:
            print("process_rx_chunk: no full symbol found in", filename)
            return
        print("process_rx_chunk: found", num_symbols, "OFDM symbols in", filename)

        #rx_SNR_dB=10
        rx_signal=rx_data
        YY2=min(num_symbols,YY-1)

        print("Number of saved symbols, YY, and YY2 are:", num_symbols, YY, YY2)
        #print(ffg)
        # Choose modulation and coding from the supported set.

        time.sleep(10)

        # Set search_length to cover the region where blocks occur.
        search_length = len(rx_signal) - block_size
        spacing_tolerance = block_size // 2  # candidates should be roughly one block apart
        refine_window = 3  # not used in this implementation but could be added for local refinement
        data_block_len = FFT_SIZE + CP_LEN
        # Set search_length to cover the region where blocks occur.
        search_length = len(rx_signal) - block_size

        print("Search length, block length (2*(nfft+cp)), and, data block length (nfft+cp) are: ", search_length, block_size)

		#sync_freq_subcarriers=np.sqrt(FFT_SIZE)*sync_freq_subcarriers

        sync_time_start = time.time()
        print("I think this function took more time!!!")
        results = synchronize_repetitive_ofdm_blocks(
                rx_signal,
                sync_freq_subcarriers,
                fft_size,
                cp_len,
                data_block_len,
                YY2,
                GG,
                sampling_rate,
                search_length=search_length,
                spacing_tolerance=spacing_tolerance)

        print("Time to search synchronization blocks (in sec) (This needs some thinking to improve)", time.time() - sync_time_start)
        #time.sleep(20)
        [sync_freq_subcarriers,rxToEstimateNoisePower1]=results["sync_and_rx_signal_for_noise_est"]

        print("Sync Freqiuency sub-carriers (first 10 values)=", sync_freq_subcarriers[:10])
        estNoisePower1=[]
        estsnr_linear=[]
        for i in range(len(rxToEstimateNoisePower1)):
            temp_noise_power, temp_h_est,snr_db, snr_linear=estimate_noise_power_and_snr_from_pilots(sync_freq_subcarriers, rxToEstimateNoisePower1[i])
            estsnr_linear.append(snr_linear)
            estNoisePower1.append(temp_noise_power)

        print("ALL Linear SNRs from all starts (SNR is not really correct). But decoding works well =", estsnr_linear)
        print("ALL Linear Noise Power from all starts (Not sure this is accurate). Again Decoding works out=", estNoisePower1)
        avsnr_linear=sum(estsnr_linear)/len(estsnr_linear)

        est_snr_db = 10 * np.log10(avsnr_linear) if avsnr_linear > 0 else -np.inf

        #est_snr_db=20

        estNoisePower1=sum(estNoisePower1)/len(estNoisePower1)

        print("Average Estimated Noise Power =", estNoisePower1)
        print("Average Estimated SNR (in dB) =", est_snr_db)

        preprocessing_time = time.time() - decoding_time_start
        print("Time to preprocess (in sec) = ", preprocessing_time)
        allMCSAndEqSym1=[]
        
        if results["candidate_starts"]:
            print("All caldidate Start and CFOs are")
            all_candidate_start_indx = results["candidate_starts"]
            all_candidate_start_indx_cfo = results["candidate_cfos"]

            print("All candidate index,", all_candidate_start_indx)
            print("All candidate CFOs,", all_candidate_start_indx_cfo)

            #print(stp1)

            '''
            allCFOs1=[]
            for i in range(len(all_candidate_start_indx)):
                tempStart1=all_candidate_start_indx[i]
                tempLocalCFOs1=cfo_estimation_from_cp(rx_signal[tempStart1:], FFT_SIZE, CP_LEN, sampling_rate)
                #print(tempLocalCFOs1)
                allCFOs1=allCFOs1+ tempLocalCFOs1
            
            eshaustiveAverageCFO1=sum(allCFOs1)/len(allCFOs1)
            print("**************** ALL CFOs computed exhaustively ***************")
            print("Average CFO computed exhaustively=", eshaustiveAverageCFO1)
            '''

            selectiveAverageCFO1=sum(all_candidate_start_indx_cfo)/len(all_candidate_start_indx_cfo)
            print("**************** ALL CFOs computed from returned indexes only ***************")
            print("Average CFO computed selectively=", selectiveAverageCFO1)

            tempL1=len(all_candidate_start_indx)
            #tempL1=10

            for i in range(tempL1):

                #cfo_estimation_from_cp(rx_signal, FFT_SIZE, CP_LEN, sampling_rate)
                #ict1
                tempStart1=all_candidate_start_indx[i]
                tempCFO1=all_candidate_start_indx_cfo[i]
                #tempCFO1=eshaustiveAverageCFO1
                tempCFO1=selectiveAverageCFO1
                #tempCFO1=0
                #print("Now we are examinining start index = ", i)
                tempIndx1=tempStart1
                tempSyncAndContSym1=rx_signal[tempIndx1:]
                tempSyncAndContSym1=tempSyncAndContSym1[:FFT_SIZE+CP_LEN]
                tempDataSym1=rx_signal[tempIndx1+FFT_SIZE+CP_LEN:]
                #tempDataSym1=rx_signal[tempIndx1+FFT_SIZE:]
                tempDataSym1=tempDataSym1[:FFT_SIZE+CP_LEN]

                print("FFT Size and CP LEN are", FFT_SIZE, CP_LEN)
                print("Sizes of Rx signal, sync and data ofdm symbols", len(rx_signal), len(tempDataSym1), len(tempSyncAndContSym1))

                if len(tempDataSym1)<FFT_SIZE+CP_LEN or len(tempSyncAndContSym1) < FFT_SIZE+CP_LEN:
                    Warning("Something is wrong (Or Rx is trimmed)")
                    continue
                if success1==1:
                    break
                
                #all_candidate_start_indx_cfo=[0]+all_candidate_start_indx_cfo
                #print("Print all CFO including no correction")
                #print(all_candidate_start_indx_cfo)

                #tempCFO1=all_candidate_start_indx_cfo[ict1]
                print("Now we are examining the following start index and CFOs: ", tempStart1 , tempCFO1)
                tempSyncAndContSym2 = correct_frequency_offset(tempSyncAndContSym1, tempCFO1, sampling_rate, 0)
                tempSyncAndContSym2=tempSyncAndContSym1
                print("Length before and after offset correction (Must be same)", len(tempSyncAndContSym1), len(tempSyncAndContSym2))
                decoded_results = decode_sync_and_control_ofdm_symbol(tempSyncAndContSym2, FFT_SIZE, CP_LEN, k1, k2, u_sync=1, u_mod=1, u_code=1)
                print("Decoded Control Information:")
                print("Estimated Modulation:", decoded_results["estimated_modulation"])
                print("Estimated Coding Rate:", decoded_results["estimated_coding"])
                print("Estimated Mod Shift:", decoded_results["mod_shift"])
                print("Estimated Code Shift:", decoded_results["code_shift"])
                print("Recall: Average Estimated SNR (in dB) =", est_snr_db)

                #print(ddr)

                est_modulation=decoded_results["estimated_modulation"]
                est_code_rate=decoded_results["estimated_coding"]
                tempDataSym2=correct_frequency_offset(tempDataSym1, tempCFO1, sampling_rate, 0)
                tempDataSym2=tempDataSym2[CP_LEN:]
                # Equalization phase
                [eq_sym,full_ch_est_values]=equalize_each_subcarrier_with_mmse_channel_estimation_ofdm(tempDataSym2, pilot_positions, pilot_value,est_snr_db)

                tempMCSandEqDict1={"Modulation": est_modulation, "Coding": est_code_rate,   "eqsym": eq_sym}
                allMCSAndEqSym1.append(tempMCSandEqDict1)

        averageMCSAndEqSym1 = group_and_average_eqsym(allMCSAndEqSym1,GG)

        if len(averageMCSAndEqSym1)>0:
            for item in averageMCSAndEqSym1:
                #print("Modulation:", item["Modulation"], "Coding:", item["Coding"],"Averaged eqsym:", item["eqsym"])
                print("Modulation:", item["Modulation"], "Coding:", item["Coding"])

                est_modulation=item["Modulation"]
                est_code_rate=item["Coding"]
                eq_sym=item["eqsym"]
                
                #plot_constellation(est_modulation, eq_sym)
                #plot_constellation(est_modulation,tempDataSym2)
                #tempDataSym2=

                #eq_sym = np.fft.fft(tempDataSym2).astype(np.complex64)

                [K,used_subcarriers,unused_subcarriers]=compute_size_of_Tx_Rx_bits_and_subcarriers(max_used_subcarriers_excluding_pilot,est_modulation,est_code_rate,input_crc_size)

                #num_subcarriers=used_subcarriers

                RU_SIZE=used_subcarriers # This is excluding pilots
                # APPLY CRC HERE
                print("RU Size=", RU_SIZE)

                # We create info bits, add CRC, encode => codeword => mod => map subcarriers => ...
                # We'll do 72 user bits => plus 16 CRC => 88 total
                user_bits= np.random.randint(0,2, K, dtype=np.uint8)
                user_with_crc= add_crc16_bits(user_bits)
                # Encode
                final_info, codeword, mod_signal, H_qc= encode_qc3_11ax(user_with_crc, RU_SIZE, est_modulation, est_code_rate)
                print("Single user: final info len=", len(final_info), " codeword len=", len(codeword), " # symbols:", len(mod_signal))

                # For example, allocate the RU block in the center of the FFT:
                #start_idx = (FFT_SIZE - RU_SIZE - len(pilotPosWithInRSCMinSCMax1)) // 2
                #RU_range = np.arange(start_idx, start_idx + RU_SIZE + len(pilotPosWithInRSCMinSCMax1))

                # Pilot positions are specified relative to the full FFT.
                # For instance, if you want pilots at absolute indices 60 and 190:
                #pilot_positions_NFFT = [60, 190]
                #pilot_value = 1 + 1j  # example pilot symbol

                data_subcarrier_positions=getDataSubcarrierIndexes1(FFT_SIZE,len(mod_signal),pilot_positions)
                data_sym=eq_sym[data_subcarrier_positions]
                txBitSize1=len(final_info)

                #tempDem=demodulate(data_sym,est_modulation)
                #est_snr_db=100

                #est_snr_db=30
                #[txBitSize1,used_subcarriers,H_3]=compute_size_of_bits_given_nfft_mcs(FFT_SIZE,modulation,desired_rate,input_crc_size)
                # decode with SNR aware
                #corr3_aware, rec3_aware = decode_qc3(rx_3, H_3, len(final_info_3), modulation, SNR_dB=SNR_dB, aware=True)
                
                corr3_aware, rec3_aware = decode_qc3_11ax(data_sym, H_qc, txBitSize1, est_modulation, SNR_dB=est_snr_db, aware=True)
                error_count, crc_ok, payload= check_crc16_bits(rec3_aware)
                print("Number of Errors in CRC bits = ", error_count)

                print("Payload size", len(payload))
                if crc_ok:
                    #print("CRC OK, final length of payload=", len(payload_no_crc))
                    print("CRC OK, final length of payload=", len(payload))
                    # compare with original info_bits if you want
                    # if you padded, remove the pad
                    success1=1
                    tempPayload1=payload
                    decoding_time = time.time() - decoding_time_start
                    print("Total Time to decode the trasmitted bits (in sec) = ", decoding_time)
                    return [tempPayload1,success1]
                else:
                    print("CRC fail => request retransmit")
                
                    tempTime1=time.time() - decoding_time_start
                    tempTimeMax=100.0
                    if tempTime1>=tempTimeMax:
                        success1=0
                        tempPayload1=[]
                        print("Time is over: Execution aborted!")
                        return [tempPayload1,success1]

                print("***************************************")

    decoding_time = time.time() - decoding_time_start
    print("Total Time to decode the trasmitted bits (in sec) = ", decoding_time)
    return [tempPayload1,success1]


def group_and_average_eqsym(dict_list,GG):
    """
    Given a list of dictionaries, each with keys:
      - "Modulation": string (e.g., "BPSK", "16QAM", etc.)
      - "Coding": float (e.g., 0.5, 0.3333, etc.)
      - "eqsym": NumPy vector (of complex numbers)
      
    Group entries by (Modulation, Coding) and average their eqsym vectors.
    
    Returns a list of dictionaries with the same keys, but where the eqsym value is the average
    of all vectors in that group.
    """
    groups = {}
    for entry in dict_list:
        # Use tuple (Modulation, Coding) as the group key.
        # If you worry about floating point precision, consider rounding Coding value.
        key = (entry["Modulation"], entry["Coding"])
        if key not in groups:
            groups[key] = []
        groups[key].append(entry["eqsym"])
    
    # Build output list: for each group, average along axis=0.
    output = []
    for key, vec_list in groups.items():
        # Stack vectors into a 2D array (each row is one vector)
        stacked = np.vstack(vec_list)
        avg_vec = np.mean(stacked, axis=0)

        mcsRepitition1=stacked.shape[0]
        print("Candidate Modulation and coding scheme (MCS) =",key)
        print("Number of Repitition with this MCS=", stacked.shape[0])

        mscRepititionThreshold1=GG
        mscRepititionThreshold1=75
        if mcsRepitition1>mscRepititionThreshold1:
            output.append({
                "Modulation": key[0],
                "Coding": key[1],
                "eqsym": avg_vec,
                "repetitions": stacked.shape[0]
            })

    print("Number of candidate MCS are", len(output))

    time.sleep(10)

    return output


def process_rx_chunk_simple(out_folder,FFT_SIZE):

    decoding_time_start = time.time()

    #FFT_SIZE=256
    A1=commonTxRxParameters(FFT_SIZE)

    TT=A1[0]
    KK=A1[1]
    YY=A1[2]
    GG=A1[3]
    k1=A1[4]
    k2=A1[5]
    subcarrier_spacing=A1[6]
    tx_sampling_rate=A1[7]
    rx_sampling_rate=A1[8]
    TX_FREQ=A1[9]
    RX_FREQ=A1[10]
    SYNC_DELAY=A1[11]
    pilot_positions=A1[12]
    pilot_value=A1[13]
    usedSCmin1=A1[14]
    usedSCmax1=A1[15]
    pilotPosWithInRSCMinSCMax1=A1[16]
    input_crc_size=A1[17]
    max_used_subcarriers_excluding_pilot=A1[18]
    CP_LEN=A1[19]

    sync_freq_subcarriers=generate_sync_freq_symbols(FFT_SIZE, k1, k2,u_sync=1)
    fft_size=FFT_SIZE
    rx_sampling_rate= FFT_SIZE* subcarrier_spacing
    cp_len=CP_LEN
    sampling_rate=rx_sampling_rate
    #GG=min(YY2,GG)
    block_size=2*(FFT_SIZE + CP_LEN)

    """
    Offline processing of a single chunk file:
      1) Read .bin file (complex64 samples).
      2) Split into OFDM-like symbols (of length CP_LEN+FFT_SIZE).
      3) For each symbol, remove CP, FFT, then do simple LS channel est on pilot positions.
         Then extract data subcarriers => demod => LSD decode => check CRC.
    For demonstration, we handle one RU (size RU_SIZE) in the lower subcarriers (e.g. subcarriers [0..RU_SIZE-1]).
    The rest are zero.
    """

    #files = [f for f in os.listdir(out_folder) if os.path.isfile(f)]
    #files = [f for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]
    files0 = [os.path.join(out_folder, f) for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]

    print("All saved Rx files are:")
    files=sorted(files0)
    print(files)

    success1=0
    tempPayload1=[]

    for filename in files:
        if success1==1:
            break
        rx_data = np.fromfile(filename, dtype=np.complex64)
        symbol_len = CP_LEN + FFT_SIZE
        num_symbols = len(rx_data) // symbol_len
        if num_symbols == 0:
            print("process_rx_chunk: no full symbol found in", filename)
            return
        print("process_rx_chunk: found", num_symbols, "OFDM symbols in", filename)

        rx_SNR_dB=10
        rx_signal=rx_data
        YY2=min(num_symbols,YY-1)
        # Choose modulation and coding from the supported set.
    
        # Set search_length to cover the region where blocks occur.
        search_length = len(rx_signal) - block_size
        spacing_tolerance = block_size // 2  # candidates should be roughly one block apart
        refine_window = 3  # not used in this implementation but could be added for local refinement
        data_block_len = FFT_SIZE + CP_LEN
        # Set search_length to cover the region where blocks occur.
        search_length = len(rx_signal) - block_size

        #sync_freq_subcarriers=np.sqrt(FFT_SIZE)*sync_freq_subcarriers

        results = synchronize_repetitive_ofdm_blocks(
                rx_signal,
                sync_freq_subcarriers,
                fft_size,
                cp_len,
                data_block_len,
                YY2,
                GG,
                sampling_rate,
                search_length=search_length,
                spacing_tolerance=spacing_tolerance
            )
        
        [sync_freq_subcarriers,rxToEstimateNoisePower1]=results["sync_and_rx_signal_for_noise_est"]

        estNoisePower1=[]
        estsnr_linear=[]
        for i in range(len(rxToEstimateNoisePower1)):
            temp_noise_power, temp_h_est,snr_db, snr_linear=estimate_noise_power_and_snr_from_pilots(sync_freq_subcarriers, rxToEstimateNoisePower1[i])
            estsnr_linear.append(snr_linear)
            estNoisePower1.append(temp_noise_power)

        print("ALL Linear SNRs from all starts (SNR is not really correct). But decoding works well =", estsnr_linear)
        print("ALL Linear Noise Power from all starts (Not sure this is accurate). Again Decoding works out=", estNoisePower1)
        avsnr_linear=sum(estsnr_linear)/len(estsnr_linear)

        est_snr_db = 10 * np.log10(avsnr_linear) if avsnr_linear > 0 else -np.inf

        #est_snr_db=20

        estNoisePower1=sum(estNoisePower1)/len(estNoisePower1)

        print("Average Estimated Noise Power =", estNoisePower1)
        print("Average Estimated SNR (in dB) =", est_snr_db)

        #print (results)
        #print(ddf)
        if results["candidate_starts"]:
            print("All caldidate Start and CFOs are")
            all_candidate_start_indx = results["candidate_starts"]
            all_candidate_start_indx_cfo = results["candidate_cfos"]

            print("All candidate index,", all_candidate_start_indx)

            for i in all_candidate_start_indx:
                print("Now we are examinining start index = ", i)
                tempIndx1=i
                tempSyncAndContSym1=rx_signal[tempIndx1:]
                tempSyncAndContSym1=tempSyncAndContSym1[:FFT_SIZE+CP_LEN]
                tempDataSym1=rx_signal[tempIndx1+FFT_SIZE+CP_LEN:]
                tempDataSym1=tempDataSym1[:FFT_SIZE+CP_LEN]

                print("FFT Size and CP LEN are", FFT_SIZE, CP_LEN)

                print("Sizes of Rx signal, sync and data ofdm symbols", len(rx_signal), len(tempDataSym1), len(tempSyncAndContSym1))

                if len(tempDataSym1)<FFT_SIZE+CP_LEN or len(tempSyncAndContSym1) < FFT_SIZE+CP_LEN:
                    continue
                if success1==1:
                    break

                all_candidate_start_indx_cfo=[0]+all_candidate_start_indx_cfo

                print("Print all CFO including no correction")
                print(all_candidate_start_indx_cfo)

                for j in all_candidate_start_indx_cfo:
                    print("Now we are examining the following start index and CFOs: ", i, j)
                    tempSyncAndContSym2 = correct_frequency_offset(tempSyncAndContSym1, j, sampling_rate, 0)
                    print("Length before and after offset correction (Must be same)", len(tempSyncAndContSym1), len(tempSyncAndContSym2))
                    decoded_results = decode_sync_and_control_ofdm_symbol(tempSyncAndContSym2, FFT_SIZE, CP_LEN, k1, k2, u_sync=1, u_mod=1, u_code=1)
                    print("Decoded Control Information:")
                    print("Estimated Modulation:", decoded_results["estimated_modulation"])
                    print("Estimated Coding Rate:", decoded_results["estimated_coding"])
                    print("Estimated Mod Shift:", decoded_results["mod_shift"])
                    print("Estimated Code Shift:", decoded_results["code_shift"])
                    
                    '''
                    
                    est_modulation=decoded_results["estimated_modulation"]
                    est_code_rate=decoded_results["estimated_coding"]
                    tempDataSym2=correct_frequency_offset(tempDataSym1, j, sampling_rate, 0)
                    tempDataSym2=tempDataSym2[CP_LEN:]

                    # Equalization phase

                    [eq_sym,full_ch_est_values]=equalize_each_subcarrier_with_mmse_channel_estimation_ofdm(tempDataSym2, pilot_positions, pilot_value,estNoisePower1)

                    #tempDataSym2=

                    #eq_sym = np.fft.fft(tempDataSym2).astype(np.complex64)

                    [K,used_subcarriers,unused_subcarriers]=compute_size_of_Tx_Rx_bits_and_subcarriers(max_used_subcarriers_excluding_pilot,est_modulation,est_code_rate,input_crc_size)
        
                    #num_subcarriers=used_subcarriers

                    RU_SIZE=used_subcarriers # This is excluding pilots
                    # APPLY CRC HERE
                    print("RU Size=", RU_SIZE)

                    # We create info bits, add CRC, encode => codeword => mod => map subcarriers => ...
                    # We'll do 72 user bits => plus 16 CRC => 88 total
                    user_bits= np.random.randint(0,2, K, dtype=np.uint8)
                    user_with_crc= add_crc16_bits(user_bits)
                    # Encode
                    final_info, codeword, mod_signal, H_qc= encode_qc3_11ax(user_with_crc, RU_SIZE, est_modulation, est_code_rate)
                    print("Single user: final info len=", len(final_info), " codeword len=", len(codeword), " # symbols:", len(mod_signal))

                    # For example, allocate the RU block in the center of the FFT:
                    #start_idx = (FFT_SIZE - RU_SIZE - len(pilotPosWithInRSCMinSCMax1)) // 2
                    #RU_range = np.arange(start_idx, start_idx + RU_SIZE + len(pilotPosWithInRSCMinSCMax1))

                    # Pilot positions are specified relative to the full FFT.
                    # For instance, if you want pilots at absolute indices 60 and 190:
                    #pilot_positions_NFFT = [60, 190]
                    #pilot_value = 1 + 1j  # example pilot symbol

                    data_subcarrier_positions=getDataSubcarrierIndexes1(FFT_SIZE,len(mod_signal),pilot_positions)
                    data_sym=eq_sym[data_subcarrier_positions]
                    txBitSize1=len(final_info)

                    #est_snr_db=100

                    #[txBitSize1,used_subcarriers,H_3]=compute_size_of_bits_given_nfft_mcs(FFT_SIZE,modulation,desired_rate,input_crc_size)
                    # decode with SNR aware
                    #corr3_aware, rec3_aware = decode_qc3(rx_3, H_3, len(final_info_3), modulation, SNR_dB=SNR_dB, aware=True)
                    corr3_aware, rec3_aware = decode_qc3_11ax(data_sym, H_qc, txBitSize1, est_modulation, SNR_dB=est_snr_db, aware=True)

                    error_count, crc_ok, payload= check_crc16_bits(rec3_aware)

                    print("Number of Errors in CRC bits = ", error_count)

                    print("Payload size", len(payload))
                    if crc_ok:
                        #print("CRC OK, final length of payload=", len(payload_no_crc))
                        print("CRC OK, final length of payload=", len(payload))
                        # compare with original info_bits if you want
                        # if you padded, remove the pad
                        success1=1
                        tempPayload1=payload
                        break
                    else:
                        print("CRC fail => request retransmit")

                    print("***************************************")
                    
                    '''
           

    decoding_time = time.time() - decoding_time_start
    #print("Total Time to decode the trasmitted bits (in sec) = ", decoding_time)
    #return [tempPayload1,success1]





def process_rx_chunk_spectrum(out_folder,FFT_SIZE):

    decoding_time_start = time.time()

    #FFT_SIZE=256
    A1=commonTxRxParameters(FFT_SIZE)

    TT=A1[0]
    KK=A1[1]
    YY=A1[2]
    GG=A1[3]
    k1=A1[4]
    k2=A1[5]
    subcarrier_spacing=A1[6]
    tx_sampling_rate=A1[7]
    rx_sampling_rate=A1[8]
    TX_FREQ=A1[9]
    RX_FREQ=A1[10]
    SYNC_DELAY=A1[11]
    pilot_positions=A1[12]
    pilot_value=A1[13]
    usedSCmin1=A1[14]
    usedSCmax1=A1[15]
    pilotPosWithInRSCMinSCMax1=A1[16]
    input_crc_size=A1[17]
    max_used_subcarriers_excluding_pilot=A1[18]
    CP_LEN=A1[19]

    sync_freq_subcarriers=generate_sync_freq_symbols(FFT_SIZE, k1, k2,u_sync=1)
    fft_size=FFT_SIZE
    rx_sampling_rate= FFT_SIZE* subcarrier_spacing
    cp_len=CP_LEN
    sampling_rate=rx_sampling_rate
    #GG=min(YY2,GG)
    block_size=2*(FFT_SIZE + CP_LEN)

    """
    Offline processing of a single chunk file:
      1) Read .bin file (complex64 samples).
      2) Split into OFDM-like symbols (of length CP_LEN+FFT_SIZE).
      3) For each symbol, remove CP, FFT, then do simple LS channel est on pilot positions.
         Then extract data subcarriers => demod => LSD decode => check CRC.
    For demonstration, we handle one RU (size RU_SIZE) in the lower subcarriers (e.g. subcarriers [0..RU_SIZE-1]).
    The rest are zero.
    """

    #files = [f for f in os.listdir(out_folder) if os.path.isfile(f)]
    #files = [f for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]
    files0 = [os.path.join(out_folder, f) for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]

    print("All saved Rx files are:")
    files=sorted(files0)
    files=files[7:14]
    print(files)

    success1=0
    tempPayload1=[]

    for filename in files:
        if success1==1:
            break
        rx_data = np.fromfile(filename, dtype=np.complex64)
        symbol_len = CP_LEN + FFT_SIZE
        num_symbols = len(rx_data) // symbol_len
        if num_symbols == 0:
            print("process_rx_chunk: no full symbol found in", filename)
            return
        print("process_rx_chunk: found", num_symbols, "OFDM symbols in", filename)

        rx_SNR_dB=10
        rx_signal=rx_data
        YY2=min(num_symbols,YY-1)
        # Choose modulation and coding from the supported set.
    
        # Set search_length to cover the region where blocks occur.
        search_length = len(rx_signal) - block_size
        spacing_tolerance = block_size // 2  # candidates should be roughly one block apart
        refine_window = 3  # not used in this implementation but could be added for local refinement
        data_block_len = FFT_SIZE + CP_LEN
        # Set search_length to cover the region where blocks occur.
        search_length = len(rx_signal) - block_size

        #sync_freq_subcarriers=np.sqrt(FFT_SIZE)*sync_freq_subcarriers

        results = synchronize_repetitive_ofdm_blocks(
                rx_signal,
                sync_freq_subcarriers,
                fft_size,
                cp_len,
                data_block_len,
                YY2,
                GG,
                sampling_rate,
                search_length=search_length,
                spacing_tolerance=spacing_tolerance
            )
        
        [sync_freq_subcarriers,rxToEstimateNoisePower1]=results["sync_and_rx_signal_for_noise_est"]

        estNoisePower1=[]    
        estsnr_linear=[]
        for i in range(len(rxToEstimateNoisePower1)):
            temp_noise_power, temp_h_est,snr_db, snr_linear=estimate_noise_power_and_snr_from_pilots(sync_freq_subcarriers, rxToEstimateNoisePower1[i])
            estsnr_linear.append(snr_linear)
            estNoisePower1.append(temp_noise_power)

        print("ALL Linear SNRs from all starts (SNR is not really correct: closer for awgn). But decoding works well =", estsnr_linear)
        print("ALL Linear Noise Power from all starts (Not sure if this is accurate). Again Decoding works out in awgn=", estNoisePower1)
        avsnr_linear=sum(estsnr_linear)/len(estsnr_linear)

        est_snr_db = 10 * np.log10(avsnr_linear) if avsnr_linear > 0 else -np.inf

        #est_snr_db=20
    
        estNoisePower1=sum(estNoisePower1)/len(estNoisePower1)

        print("Average Estimated Noise Power =", estNoisePower1)
        print("Average Estimated SNR (in dB) =", est_snr_db)

        #print (results)
        #print(ddf)
        if results["candidate_starts"]:
            print("All caldidate Start and CFOs are")
            all_candidate_start_indx = results["candidate_starts"]
            all_candidate_start_indx_cfo = results["candidate_cfos"]

            print("All candidate index,", all_candidate_start_indx)

            for i in all_candidate_start_indx:
                print("Now we are examinining start index = ", i)
                tempIndx1=i
                tempSyncAndContSym1=rx_signal[tempIndx1:]
                tempSyncAndContSym1=tempSyncAndContSym1[:FFT_SIZE+CP_LEN]
                tempDataSym1=rx_signal[tempIndx1+FFT_SIZE+CP_LEN:]
                tempDataSym1=tempDataSym1[:FFT_SIZE+CP_LEN]

                # It is already shited where center = 0
				#plt.figure()
				#plt.plot(tempDataSym1.real)
				#plt.plot(np.abs(np.fft.fftshift(tempDataSym1)))
				#plt.title("Magnitude of freq_full (DC at index 0)")
				#plt.xlabel("Subcarrier Index")
				#plt.ylabel("Magnitude")
				#plt.grid(True)
				#plt.show()
                
                print("FFT Size and CP LEN are", FFT_SIZE, CP_LEN)

                plt.figure()
                plt.plot(tempDataSym1.real)
                
                plt.xlabel("Time domain")
                plt.ylabel("Real amplitued")
                plt.grid(True)
                plt.show()
                print("Sizes of Rx signal, sync and data ofdm symbols", len(rx_signal), len(tempDataSym1), len(tempSyncAndContSym1))

                if len(tempDataSym1)<FFT_SIZE+CP_LEN or len(tempSyncAndContSym1) < FFT_SIZE+CP_LEN:
                    continue
                if success1==1:
                    break

                all_candidate_start_indx_cfo=[0]+all_candidate_start_indx_cfo

                print("Print all CFO including no correction")
                print(all_candidate_start_indx_cfo)

                for j in all_candidate_start_indx_cfo:
                    print("Now we are examining the following start index and CFOs: ", i, j)
                    tempSyncAndContSym2 = correct_frequency_offset(tempSyncAndContSym1, j, sampling_rate, 0)
                    print("Length before and after offset correction (Must be same)", len(tempSyncAndContSym1), len(tempSyncAndContSym2))
                    decoded_results = decode_sync_and_control_ofdm_symbol(tempSyncAndContSym2, FFT_SIZE, CP_LEN, k1, k2, u_sync=1, u_mod=1, u_code=1)
                    print("Decoded Control Information:")
                    print("Estimated Modulation:", decoded_results["estimated_modulation"])
                    print("Estimated Coding Rate:", decoded_results["estimated_coding"])
                    print("Estimated Mod Shift:", decoded_results["mod_shift"])
                    print("Estimated Code Shift:", decoded_results["code_shift"])



	

'''
def estimate_noise_power_and_snr_from_pilots_v2(pilots, received):
    """
    Estimate noise power from pilot signals when the channel is unknown.
    
    Also returns an estimated channel (assumed constant) and the SNR (in dB).
    
    Parameters:
      pilots   : numpy array of transmitted pilot symbols (complex), shape (N,)
      received : numpy array of corresponding received pilot symbols (complex), shape (N,)
    
    Returns:
      noise_power : Estimated noise power (average squared error)
      h_est       : Estimated channel (scalar)
      snr_db      : Estimated SNR in dB (signal power to noise power ratio)
    """
    # Avoid division by zero by using only nonzero pilots
    valid_indices = np.nonzero(pilots)[0]
    if valid_indices.size == 0:
        raise ValueError("No valid (nonzero) pilot values available for channel estimation.")
    
    # Estimate channel per pilot and average
    h_estimates = received[valid_indices] / pilots[valid_indices]
    h_est = np.mean(h_estimates)
    
    # Compute error between received and expected (using estimated channel)
    errors = received - h_est * pilots
    noise_power = np.mean(np.abs(errors)**2)
    
    # For unit-magnitude pilots, pilot power is 1 so signal power is |h_est|^2.
    estimated_signal_power = np.abs(h_est)**2
    snr_linear = estimated_signal_power / noise_power if noise_power != 0 else np.inf
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
    
    return noise_power, h_est, snr_db, snr_linear

'''

def estimate_noise_power_and_snr_from_pilots(pilots, received):
    """
    Estimate the noise power from pilot signals when the channel is unknown,
    and compute the estimated SNR.
    
    Parameters:
      pilots   : numpy array of transmitted pilot symbols (complex), shape (N,)
      received : numpy array of corresponding received pilot symbols (complex), shape (N,)
    
    Returns:
      noise_power : Estimated noise power (average error power)
      h_est       : Estimated channel (assumed constant over pilots)
      snr_db      : Estimated SNR in dB (signal power to noise power ratio)
    """
    # Select valid indices (avoid division by zero)
    valid_indices = np.nonzero(pilots)[0]
    if valid_indices.size == 0:
        raise ValueError("No valid (nonzero) pilot values available for channel estimation.")
    
    # Estimate channel for each valid pilot: h_i = y_i / x_i
    h_estimates = received[valid_indices] / pilots[valid_indices]
    print("Channel estimate from received signal and sync signal: No info on SNR (i.e., simple division, first 10 values)=", h_estimates[:10])
    # Compute the average channel estimate
    h_est = np.mean(h_estimates)
    
    print("Average estimated channel (including power scaling at Tx)=", h_est)
    # Compute errors between received pilots and the estimated signal
    errors = received - h_est * pilots
    # Estimate noise power as the average squared magnitude of the error
    noise_power = np.mean(np.abs(errors)**2)
    
    # Estimate signal power as |h_est|^2 multiplied by the average pilot power.
    pilot_power = np.mean(np.abs(pilots)**2)
    estimated_signal_power = np.abs(h_est)**2 * pilot_power
    
    # Compute SNR in linear scale and then convert to dB
    if noise_power == 0:
        snr_linear = np.inf
    else:
        snr_linear = estimated_signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
    
    return noise_power, h_est, snr_db, snr_linear


'''
def estimate_noise_power_from_pilots(pilots, received):
    """
    Estimate the noise power from pilot signals when the channel is unknown.
    
    Parameters:
      pilots   : numpy array of transmitted pilot symbols (complex), shape (N,)
      received : numpy array of corresponding received pilot symbols (complex), shape (N,)
    
    Returns:
      noise_power : Estimated noise power (average error power)
      h_est       : Estimated channel (assumed constant over pilots)
    """
    # Avoid division by zero by selecting indices where the pilot is not zero
    valid_indices = np.nonzero(pilots)[0]
    if valid_indices.size == 0:
        raise ValueError("No valid (nonzero) pilot values available for channel estimation.")
    
    # Estimate the channel for each valid pilot: h_i = y_i / x_i
    h_estimates = received[valid_indices] / pilots[valid_indices]
    # Compute a single channel estimate by averaging
    h_est = np.mean(h_estimates)
    
    # Compute the error between the received pilot and the estimated signal
    errors = received - h_est * pilots
    # Noise power is the average power (squared magnitude) of these errors
    noise_power = np.mean(np.abs(errors)**2)
    
    return noise_power, h_est
'''

def interpolate_complex_values(known_indices, known_values, max_index):
    """
    Linearly interpolate missing complex values.

    Parameters:
      known_indices : list or array of indices where values are known (must be in increasing order)
      known_values  : list or array of known complex numbers corresponding to known_indices
      max_index     : integer, the maximum index of the complete vector

    Returns:
      full_values   : numpy array of complex numbers from index 0 to max_index with missing values interpolated.
    """
    # Create the full array of indices
    full_indices = np.arange(0, max_index + 1)
    
    # Separate the real and imaginary parts of the known values
    real_known = np.real(known_values)
    imag_known = np.imag(known_values)
    
    # Use np.interp to interpolate the real and imaginary parts separately
    real_interp = np.interp(full_indices, known_indices, real_known)
    imag_interp = np.interp(full_indices, known_indices, imag_known)
    
    # Combine the interpolated real and imaginary parts back into complex numbers
    full_values = real_interp + 1j * imag_interp
    return full_values


def equalize_each_subcarrier_with_mmse_channel_estimation_ofdm(time_symbol, pilot_positions, pilot_value,snr_dB):
    
    snr_linear=10**(snr_dB/10)
    freq_sym = np.fft.fft(time_symbol)
    
    """
    MMSE Estimator for each subcarrier
    """
    ch_est_values=[]
    #sigma_h=noise_power
    pilot_vals = []
    for p in pilot_positions:
        #print(p)
        #num1=sigma_h*np.conjugate(pilot_value)
        #print(num1)
        #demom1=(sigma_h*abs(pilot_value*np.conjugate(pilot_value)) + noise_power)
        #print(demom1)
        #print(len(freq_sym))
        #print(freq_sym[p])
        #temp1=freq_sym[p]*(num1/demom1)
        #temp1=freq_sym[p]/pilot_value
        temp1=(snr_linear/(snr_linear+1))*(freq_sym[p]/pilot_value)
        ch_est_values.append(temp1)
    
    # Example:
    # Given complex vector values:
    # x[0] = 0.1+0.2j, x[2] = 0.8+0.6j, x[5] = 0.6+0.9j, x[7] = 0.99+0.2j
    known_indices = pilot_positions # [0, 2, 5, 7]
    known_values  = ch_est_values #np.array([0.1+0.2j, 0.8+0.6j, 0.6+0.9j, 0.99+0.2j])
    max_index = len(freq_sym)  # Complete the vector from x[0] to x[7]

    print(len(freq_sym))
    print(max_index)

    # Interpolate to fill in missing indices
    full_ch_est_values = interpolate_complex_values(known_indices, known_values, max_index)
    eq_sym=freq_sym
    for i in range(len(full_ch_est_values)-1):
        h_est=full_ch_est_values[i]
        if abs(h_est)!=0:
            eq_sym[i]=eq_sym[i]/h_est

    print("Interpolated vector:")
    #print(full_ch_est_values)
    return [eq_sym,full_ch_est_values]


def compute_size_of_bits_given_nfft_mcs(FFT_SIZE,modulation,desired_rate,input_crc_size):

    b_dict={'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
    if modulation not in b_dict:
        raise ValueError("Mod not supported!")
    b=b_dict[modulation]

    base_mat = None
    for r,mat in TOY_80211AX_MATRICES.items():
        if abs(r - desired_rate)<1e-3:
            base_mat= mat
            break
    if base_mat is None:
        raise ValueError(f"No toy base matrix found for code_rate={desired_rate}.")

    #global SMALL_R12_BASE
    #base_mat = np.array(SMALL_R12_BASE, dtype=int)  # shape (4,8)
    #base_mat= np.array(base_mat, dtype=int)
    m, n= base_mat.shape  # e.g. (12,24) for rate=1/2

    num_subcarriers=FFT_SIZE
    # total coded bits T
    T=num_subcarriers*b
    # We do floor division for z:
    z=T//n
    # final codeword length
    N=n*z
    # If N not multiple of b, we further reduce z so that N is multiple of b
    while (N % b)!=0 or (N % input_crc_size)!=0:
        z-=1
        if z<=0:
            raise ValueError("Cannot find z>0 such that N is multiple of b.")
        N=n*z
    # Now we have N <= T and N multiple of b.
    # subcarriers used
    used_subcarriers=N//b
    unused_subcarriers=(T//b)-used_subcarriers

    print("Used sub-carriers=", used_subcarriers)
    print("Unused sub-carriers=", unused_subcarriers)
    # info bits
    K=(n-m)*z

    tempInfoBits1= np.random.randint(0, 2, K-input_crc_size).astype(np.uint8)
    #tempInfoBits1=trueInfoBits1
    tempct1=-1
    tempct1=tempct1+1
    bits_with_crc = add_crc16_bits(tempInfoBits1)
    print("Transmitting total bits:", len(bits_with_crc), "bits (with CRC)")

    # Now also do the new "standard-like" approach:
    #final_info_3, codeword_3, modsig_3, H_3 = encode_qc3(bits_with_crc, num_subcarriers, modulation, SNR_dB=None)
    #final_info_3, codeword_3, modsig_3, H_3 = encode_qc3_11ax(bits_with_crc, num_subcarriers, modulation, desired_rate)
    H_3=encode_qc3_11ax_for_parameters(num_subcarriers, modulation, desired_rate)

    return [K,used_subcarriers,H_3]



def encode_qc3_11ax_for_parameters(num_subcarriers, modulation, code_rate):
    """
    A single function that:
      1) picks the base matrix for code_rate from TOY_80211AX_MATRICES,
      2) computes T= num_subcarriers*b (b= bits/symbol),
      3) picks L= floor(T/n),
      4) builds H => shape (mL, nL), G => [I|A^T],
      5) pad/truncate info_bits => length= K= (n-m)*L
      6) encode => modulate => return (final_info_bits, codeword, mod_sig, H).
    """
    # get b
    mod_dict= {'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
    if modulation not in mod_dict:
        raise ValueError("encode_qc3_11ax: unsupported mod " + modulation)
    b = mod_dict[modulation]
    # pick base matrix from dictionary
    if code_rate not in TOY_80211AX_MATRICES:
        raise ValueError("No base matrix found for code_rate=" + str(code_rate))
    base_mat = TOY_80211AX_MATRICES[code_rate]
    m,n = base_mat.shape
    
    T= num_subcarriers*b
    L= T// n
    if L<=0:
        raise ValueError("encode_qc3_11ax: not enough subcarriers to form code.")
    # codeword length => n*L
    # info => (n-m)*L
    K= (n- m)* L
    if K<=0:
        raise ValueError("encode_qc3_11ax: no info bits => K=0.")
    
    # build H
    H_qc= np.zeros((m*L, n*L), dtype=np.uint8)
    from copy import deepcopy
    for i in range(m):
        for j in range(n):
            val= base_mat[i,j]
            if val>=0:
                block= np.eye(L,dtype=np.uint8)
                block= np.roll(block, val%L, axis=1)
                H_qc[i*L:(i+1)*L, j*L:(j+1)*L]= block
    
    return H_qc

#######################################
# Part 2: Custom LDPC for Code Rate = 0.75, QC-LDPC Tools, and Mod/Demod
#######################################

def generate_TOY_BASE_12x72():
    """
    Build a "toy" 12x72 base matrix for rate=5/6. 
    The left 60 columns use a small repeated pattern. 
    The right 12 columns have SHIFT=0 on the diagonal (row i => col(60 + i) = 0)
    so that the final code is systematic in a noise-free scenario.
    """
    # We'll define an 8-column pattern of SHIFT or -1:
    pattern8 = [0, -1, 2, -1, 1, -1, 3, -1]
    # We'll repeat that pattern 7 times => 56 columns, then add 4 more => total 60 columns.
    left60 = pattern8 * 7 + [0, -1, 2, -1]  # total 60 columns
    
    # Now build 12 rows. For row i, the rightmost 12 columns are all -1 except col(60 + i) => 0
    arr72 = []
    for i in range(12):
        # Right block => 12 columns, SHIFT=0 at index i
        right12 = [-1]*12
        right12[i] = 0
        row = left60 + right12
        arr72.append(row)
    
    return np.array(arr72, dtype=int)

TOY_BASE_12x72 = generate_TOY_BASE_12x72()


TOY_BASE_12x36 = np.array([
# Each row has 36 columns: 
# - The first 24 columns are a repeated pattern of small SHIFTs in {0,1,2,3} or -1
# - The last 12 columns: SHIFT=0 on the diagonal => col(24 + row_i) = 0, rest = -1

# Row0 => col24=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,  
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],

# Row1 => col25=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],

# Row2 => col26=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],

# Row3 => col27=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1 ],

# Row4 => col28=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1 ],

# Row5 => col29=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1 ],

# Row6 => col30=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1 ],

# Row7 => col31=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1 ],

# Row8 => col32=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1 ],

# Row9 => col33=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1 ],

# Row10 => col34=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1 ],

# Row11 => col35=0
[  0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
   0, -1,  2, -1,  1, -1,  0, -1,  2, -1,  1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
], dtype=int)

SMALL_R12_BASE = [
    # We want m=4, n=8 => nominal rate= (8-4)/8=0.5
    # The left block has size 4×(8-4)=4
    # We fill each row in the left block with random shifts in [0,z-1] or -1 with some density
    # We'll do that automatically in code, or provide a minimal example:
    # For demonstration, here's a minimal "hard-coded" example:
    
    # left half is 4 columns, right half is 4 columns
    # row i => SHIFT=0 at col i in the right half
    # -1 => zero block,  x => shift
    # We'll just fill the left half with some small shifts (0..3) to keep it consistent
    # You can randomize it in code if you prefer.
    
    [ 0,  1, -1,  2,    0, -1, -1, -1],  # row0 => SHIFT=0 at col4
    [ 2, -1,  0, -1,   -1,  0, -1, -1],  # row1 => SHIFT=0 at col5
    [ 1, -1,  3, -1,   -1, -1,  0, -1],  # row2 => SHIFT=0 at col6
    [-1,  2, -1,  1,   -1, -1, -1,  0]   # row3 => SHIFT=0 at col7
]

SMALL_R12_BASE0 = np.array(SMALL_R12_BASE, dtype=int)


def generate_toy_base_72_rate13():
    """
    Generate a toy base matrix for a 72-column QC-LDPC code with rate = 1/3.
    For rate = 1/3, we need (n-m)/n = 1/3 → m = 72 - 72/3 = 48.
    
    Structure:
      - The left block has (n-m) = 24 columns.
      - The right block has m = 48 columns.
      - In the left block, we use a repeated pattern of 12 numbers repeated twice.
      - In the right block, for each row i (0 <= i < 48), all entries are -1 except
        that at position (i) (within the right block), we set 0.
      
    Returns a numpy array of shape (48, 72).
    """
    m = 48
    n = 72
    left_cols = n - m  # 72-48 = 24
    # Define a pattern for the left block (length 12) and repeat it twice:
    pattern = [0, 1, -1, 2, -1, 1, 0, -1, 2, -1, 1, -1]
    # To fill 24 columns, repeat pattern twice:
    left_pattern = pattern * 2   # length 24
    
    base = []
    for i in range(m):  # for each of the 48 rows
        # left block: use the same repeated pattern for every row
        row_left = left_pattern.copy()
        # right block: length = 48, all -1 except at index i (within the right block) set to 0.
        row_right = [-1] * m
        # Note: i must be less than m (i from 0 to 47) so that position i in the right block is valid.
        row_right[i] = 0
        row = row_left + row_right  # total length = 24+48 = 72
        base.append(row)
    return np.array(base, dtype=int)

def generate_toy_base_72_rate16():
    """
    Generate a toy base matrix for a 72-column QC-LDPC code with rate = 1/6.
    For rate = 1/6, we need (n-m)/n = 1/6 → m = 72 - 72/6 = 72 - 12 = 60.
    
    Structure:
      - The left block has (n-m) = 12 columns.
      - The right block has m = 60 columns.
      - In the left block, we use a fixed pattern of 12 numbers (for example).
      - In the right block, for each row i (0 <= i < 60), all entries are -1 except
        that at position (i) (within the right block) we set 0.
      
    Returns a numpy array of shape (60, 72).
    """
    m = 60
    n = 72
    left_cols = n - m  # 72-60 = 12
    # Define a pattern for the left block (length 12):
    left_pattern = [0, -1, 2, -1, 1, -1, 0, -1, 2, -1, 1, -1]
    
    base = []
    for i in range(m):  # for each of the 60 rows
        row_left = left_pattern.copy()  # length 12
        # right block: length = 60, all -1 except at index i set to 0.
        row_right = [-1] * m
        if i < m:
            row_right[i] = 0
        row = row_left + row_right  # total length = 12+60 = 72
        base.append(row)
    return np.array(base, dtype=int)


def generate_toy_base_72_rate112():
    """
    Generate a toy base matrix for a 72-column QC-LDPC code with rate = 1/12.
    
    For rate = 1/12:
      - n = 72 columns.
      - (n - m) = 6, so m = 72 - 6 = 66 rows.
      
    Structure:
      - Left block (columns 0 to 5): fixed pattern of length 6, e.g. [0, -1, 2, -1, 1, -1].
      - Right block (columns 6 to 71): for each row i (0 ≤ i < 66), all entries are -1 
        except the entry at index i (i.e. column (6 + i)) is set to 0.
        
    Returns:
      A numpy array of shape (66, 72) representing the base matrix.
    """
    m = 66
    n = 72
    left_cols = n - m  # = 72 - 66 = 6
    
    # Define the fixed pattern for the left block of length 6:
    left_pattern = [0, -1, 2, -1, 1, -1]
    
    base = []
    for i in range(m):  # i = 0, 1, ..., 65
        # For every row, the left block is identical:
        row_left = left_pattern.copy()  # length = 6
        # The right block: length = 66, all -1 except position i is 0.
        row_right = [-1] * m  # m = 66
        row_right[i] = 0
        # Concatenate to form a full row of length 6 + 66 = 72:
        row = row_left + row_right
        base.append(row)
    return np.array(base, dtype=int)



import numpy as np

def generate_toy_base_72_rate124():
    """
    Generate a toy base matrix for a 72-column QC-LDPC code with rate = 1/24.
    
    For rate = 1/24, we need (n - m)/n = 1/24.
    With n = 72, then n - m = 72/24 = 3, so m = 72 - 3 = 69.
    
    Structure:
      - Left block (columns 0 to 2): a fixed pattern of 3 numbers, e.g. [0, -1, 2].
      - Right block (columns 3 to 71): 69 columns. In each row i (0 <= i < 69),
        all entries are -1 except that the entry at index i (within the right block)
        is set to 0.
    
    Returns:
      A numpy array of shape (69, 72).
    """
    m = 69
    n = 72
    left_cols = n - m  # = 3
    # Define the fixed pattern for the left block:
    left_pattern = [0, -1, 2]
    
    base = []
    for i in range(m):  # i = 0, 1, …, 68
        # For every row, the left block is the same:
        row_left = left_pattern.copy()  # length 3
        # Build the right block: 69 columns, all -1 except at index i:
        row_right = [-1] * m  # m = 69
        row_right[i] = 0      # Set the diagonal entry to 0
        # Concatenate left and right blocks to form a row of length 3 + 69 = 72:
        row = row_left + row_right
        base.append(row)
    return np.array(base, dtype=int)


TOY_BASE_12_RATE13=generate_toy_base_72_rate13()
TOY_BASE_12_RATE16=generate_toy_base_72_rate16()
TOY_BASE_12_RATE112=generate_toy_base_72_rate112()
#TOY_BASE_12_RATE124=generate_toy_base_72_rate124()


#############################################################################
# 2) The dictionary picking each matrix for rates: 1/2, 2/3, 3/4, 5/6
#############################################################################
TOY_80211AX_MATRICES = {
  0.0833: TOY_BASE_12_RATE112,  # Rate=1/3
  0.1666:    TOY_BASE_12_RATE16, # Rate=1/6 #
  0.3333: TOY_BASE_12_RATE13,  # Rate=1/3
  0.5:    SMALL_R12_BASE0, # TOY_BASE_12x24, #
  0.6667: TOY_BASE_12x36,  # ~2/3
  0.8333: TOY_BASE_12x72,  # ~5/6
}


def lookUpTableSubCarrierMCS(tempSNR1,tempSubCarrier1):

    allNoSubCarriers1=[100,5000]
    allCodeRate1=[0.5,0.6667,0.8333,0.3333,0.1666,0.0833]
    allModulation1=["BPSK","QPSK",'16QAM',"64QAM"]
    
    if (tempSubCarrier1>=allNoSubCarriers1[0]) and (tempSubCarrier1<=allNoSubCarriers1[1]):
        if tempSNR1>=40:
            tempModulation1=allModulation1[3]
            temoCodeRate1=allCodeRate1[2]
        elif tempSNR1>=35:
            tempModulation1=allModulation1[3]
            temoCodeRate1=allCodeRate1[1]
        elif tempSNR1>=30:
            tempModulation1=allModulation1[2] # This is modified to [2][2] since the effective data rate is lower of we go to [3], [0]
            temoCodeRate1=allCodeRate1[2]
        elif tempSNR1>=25:
            tempModulation1=allModulation1[2]
            temoCodeRate1=allCodeRate1[2]
        elif tempSNR1>=22:
            tempModulation1=allModulation1[2]
            temoCodeRate1=allCodeRate1[1]
        elif tempSNR1>=18:
            tempModulation1=allModulation1[2]
            temoCodeRate1=allCodeRate1[0]
        elif tempSNR1>=14:
            tempModulation1=allModulation1[1]
            temoCodeRate1=allCodeRate1[2]
        elif tempSNR1>=9:
            tempModulation1=allModulation1[1]
            temoCodeRate1=allCodeRate1[1]
        elif tempSNR1>=5:
            tempModulation1=allModulation1[1]
            temoCodeRate1=allCodeRate1[0]
        elif tempSNR1>=2:
            tempModulation1=allModulation1[1]
            temoCodeRate1=allCodeRate1[3]
        elif tempSNR1>=-1:
            tempModulation1=allModulation1[1]
            temoCodeRate1=allCodeRate1[4]
        else:
            tempModulation1=allModulation1[1]
            temoCodeRate1=allCodeRate1[5]

    A1=[tempModulation1,temoCodeRate1]
    return A1

def pilotSubCarrierIndexes1(NFFT):
    tempVec1=[]
    gap1=14
    for i in range(1,NFFT):
        if gap1*i<=NFFT:
            tempVec1.append(gap1*i)
        else:
            break
    return tempVec1

def usedSubCarrierIndexes1(NFFT):
    tempVec1=[]
    if NFFT==256:
        usedSCmin1=-117+int(NFFT/2)
        usedSCmax1=116+int(NFFT/2)-1
    if NFFT==512:
        usedSCmin1=-235+int(NFFT/2)
        usedSCmax1=234+int(NFFT/2)-1
    if NFFT==1024:
        usedSCmin1=-469+int(NFFT/2)
        usedSCmax1=468+int(NFFT/2)-1
    if NFFT==2048:
        usedSCmin1=-937+int(NFFT/2)
        usedSCmax1=936+int(NFFT/2)-1
    if NFFT==4096:
        usedSCmin1=-1873+int(NFFT/2)
        usedSCmax1=1872+int(NFFT/2)-1

    return [usedSCmin1,usedSCmax1]


def design_rc_filter(span, sps, rolloff):
    """
    Design a raised cosine filter with given span (in symbols), samples per symbol (sps),
    and rolloff factor.
    """
    N = span * sps + 1
    taps = np.zeros(N, dtype=np.float32)
    mid = N // 2
    for n in range(N):
        t = (n - mid) / sps
        if abs(t) < 1e-6:
            taps[n] = 1.0
        elif abs(abs(t) - 1/(4*rolloff)) < 1e-6:
            taps[n] = (rolloff/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*rolloff)) +
                                              (1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
        else:
            numerator = np.sin(np.pi*t*(1-rolloff)) + 4*rolloff*t*np.cos(np.pi*t*(1+rolloff))
            denominator = np.pi*t*(1 - (4*rolloff*t)**2)
            taps[n] = numerator/denominator
    taps /= np.sum(taps)
    return taps

def generate_sync_preamble(fft_size, cp_len, rc_taps):
    """
    Generate a sync preamble by using an all-ones frequency-domain vector, IFFT,
    CP addition, and an RC filter.
    """
    sync_freq = np.ones(fft_size, dtype=np.complex64)
    sync_freq_shifted = np.fft.ifftshift(sync_freq)
    time_domain = np.fft.ifft(sync_freq_shifted).astype(np.complex64)
    cp = time_domain[-cp_len:]
    sync_symbol = np.concatenate((cp, time_domain))
    #sync_waveform = np.convolve(sync_symbol, rc_taps, mode='full').astype(np.complex64)
    sync_waveform=sync_symbol
    return sync_waveform


def generate_zc_sequence(Nzc, u=1):
    """
    Generate a Zadoff–Chu sequence of length Nzc with root u.
    x[n] = exp(-j * pi * u * n*(n+1)/Nzc) for n = 0, 1, ..., Nzc-1.
    """
    n = np.arange(Nzc)
    x = np.exp(-1j * np.pi * u * n * (n+1) / Nzc)
    return x.astype(np.complex64)

def generate_sync_and_control_symbol(fft_size, cp_len, k1=5, k2=5, u=1):
    """
    Generate one OFDM symbol that contains both a synchronization sequence and a control signal.
    
    Parameters:
      fft_size : Total number of FFT points.
      cp_len   : Length of cyclic prefix.
      k1       : Number of bits allocated for the modulation selection (for control).
      k2       : Number of bits allocated for the coding rate selection (for control).
      u        : Zadoff–Chu root (default=1).
      
    Process:
      1) Reserve (k1+k2) subcarriers for control and use the remaining subcarriers for sync.
         For example, if fft_size=128 and k1+k2=10, then 118 subcarriers carry the sync.
      2) Generate a Zadoff–Chu sequence of length = fft_size – (k1+k2) for synchronization.
      3) Generate a random control bit vector of length (k1+k2) and modulate it with QPSK.
      4) Form a frequency-domain vector of length fft_size:
           - The first (fft_size – (k1+k2)) subcarriers carry the ZC sync sequence.
           - The remaining (k1+k2) subcarriers carry the QPSK control symbols.
      5) Convert to time domain by applying ifftshift then IFFT.
      6) Prepend a cyclic prefix of length cp_len.
      
    Returns:
      time_domain_symbol : The complete OFDM symbol (with CP) in time domain.
      freq_domain        : The constructed frequency-domain vector (for debugging/plotting).
      control_info       : A tuple (control_bits, qpsk_symbols) of the control data.
    """
    total_control = k1 + k2
    sync_length = fft_size - total_control  # number of subcarriers for sync
    
    # Generate sync sequence using Zadoff–Chu
    zc_seq = generate_zc_sequence(sync_length, u=u)
    
    # Generate random control bits of length total_control
    control_bits = np.random.randint(0, 2, total_control, dtype=np.uint8)
    # Ensure even number of bits for QPSK; if odd, pad one 0.
    if len(control_bits) % 2 != 0:
        control_bits = np.concatenate([control_bits, np.zeros(1, dtype=np.uint8)])
    
    # QPSK modulation: mapping (Gray coded)
    qpsk_symbols = []
    const = 1 / np.sqrt(2)
    for i in range(0, len(control_bits), 2):
        b0 = control_bits[i]
        b1 = control_bits[i+1]
        if b0 == 0 and b1 == 0:
            sym = (1 + 1j) * const
        elif b0 == 0 and b1 == 1:
            sym = (1 - 1j) * const
        elif b0 == 1 and b1 == 1:
            sym = (-1 - 1j) * const
        else:  # b0==1 and b1==0
            sym = (-1 + 1j) * const
        qpsk_symbols.append(sym)
    qpsk_symbols = np.array(qpsk_symbols, dtype=np.complex64)
    
    # Build the full frequency-domain vector (of length fft_size).
    freq_domain = np.zeros(fft_size, dtype=np.complex64)
    # Place the sync sequence in the first sync_length subcarriers.
    freq_domain[:sync_length] = zc_seq
    # Place the control symbols in the remaining subcarriers.
    freq_domain[sync_length:] = qpsk_symbols
    
    # Convert to time domain.
    # If the freq_domain vector is already arranged with data centered,
    # we need to use ifftshift before ifft.
    time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain)).astype(np.complex64)
    
    # Add cyclic prefix.
    cp = time_domain[-cp_len:]
    time_domain_symbol = np.concatenate((cp, time_domain))
    
    control_info = (control_bits, qpsk_symbols)
    return time_domain_symbol, freq_domain, control_info



def generate_sync_and_control_ofdm_symbol(fft_size, cp_len, k1, k2,
                                            selected_modulation, selected_coding,
                                            u_sync=1, u_mod=1, u_code=1):
    """
    Generate one OFDM symbol that carries both a sync sequence and a control field,
    where the control field is generated using cyclically shifted Zadoff–Chu sequences.
    
    Parameters:
      fft_size : int
          Total FFT size.
      cp_len : int
          Cyclic prefix length.
      k1 : int
          Number of bits reserved for modulation control.
      k2 : int
          Number of bits reserved for coding control.
      selected_modulation : str
          One of: "BPSK", "QPSK", "16QAM", "64QAM", "256QAM".
      selected_coding : float
          One of: 0.5, 0.6667, 0.8333.
      u_sync : int, default 1
          Zadoff–Chu root for sync portion.
      u_mod : int, default 1
          Zadoff–Chu root for modulation control sequence.
      u_code : int, default 1
          Zadoff–Chu root for coding control sequence.
          
    Process:
      1) The control field uses a total of (k1+k2) subcarriers.
         (For example, if k1=5 and k2=5, then 10 subcarriers are reserved for control.)
      2) The remaining subcarriers (fft_size - (k1+k2)) are used for the sync sequence.
          The sync sequence is a Zadoff–Chu sequence of length = fft_size - (k1+k2).
      3) For the modulation control part (length = k1):
          - A base Zadoff–Chu sequence of length k1 is generated.
          - A cyclic shift is computed as:  
                shift_mod = round( relative_mod * (2**k1 - 1) )
            where relative_mod is a relative value in [0,1] chosen for the given modulation.
      4) For the coding control part (length = k2):
          - A base Zadoff–Chu sequence of length k2 is generated.
          - A cyclic shift is computed as:  
                shift_code = round( relative_code * (2**k2 - 1) )
            where relative_code is chosen for the given coding rate.
      5) The two shifted sequences are concatenated to form the control field.
      6) A full frequency-domain vector of length fft_size is built:
          - The first (fft_size - (k1+k2)) subcarriers are filled with the sync sequence.
          - The remaining (k1+k2) subcarriers are filled with the control field.
      7) The frequency-domain vector is converted to time domain by applying ifftshift then ifft,
          and a cyclic prefix is prepended.
    
    Returns:
      time_domain_symbol : np.ndarray (complex64)
          The resulting OFDM symbol with CP.
      freq_domain : np.ndarray (complex64)
          The constructed frequency-domain vector (unshifted).
      control_field : np.ndarray (complex64)
          The concatenated control field (after cyclic shifts).
      mcs_info : tuple
          (selected_modulation, selected_coding, shift_mod, shift_code)
    """
    # Total control subcarrier count:
    total_control = k1 + k2
    sync_length = fft_size - total_control
    if sync_length <= 0:
        raise ValueError("FFT size too small for given control subcarrier allocation.")
    
    # Generate sync portion using a Zadoff–Chu sequence (length = sync_length)
    n_sync = np.arange(sync_length)
    zc_sync = np.exp(-1j * np.pi * u_sync * n_sync * (n_sync + 1) / sync_length).astype(np.complex64)
    
    #print(np.abs(zc_sync))
    # It is already shited where center = 0
    #plt.figure()
    #plt.plot(np.abs(zc_sync))
    #plt.title("Magnitude of freq_full (DC at index 0)")
    #plt.xlabel("Subcarrier Index")
    #plt.ylabel("Magnitude")
    #plt.grid(True)
    #plt.show()
    #print(ffd)

    # Generate base ZC sequences for control:
    def generate_zc_seq(length, u):
        n = np.arange(length)
        return np.exp(-1j * np.pi * u * n * (n+1) / length).astype(np.complex64)
    
    base_mod_seq = generate_zc_seq(k1, u_mod)
    base_code_seq = generate_zc_seq(k2, u_code)
    
    # Define relative mapping values (between 0 and 1) for modulation and coding.
    # These values can be chosen to maximize separation between the candidates.
    # For example, if k1=5, there are 32 possibilities. Here we choose a few representative ones.
    # To generalize, we can define dictionaries that depend on k1 and k2.
    # For instance, we might choose:
    mod_relative = {        
        "QPSK":   0.00,
        "16QAM":  0.25,
        "64QAM":  0.50,
        "256QAM": 0.75
    }
    code_relative = {0.0833:0.0, 0.1666: 0.166, 0.3333: 0.333,
        0.5:    0.5,
        0.6667: 0.666,
        0.8333: 0.833
    }


    #"BPSK":   0.01,
    if selected_modulation not in mod_relative:
        raise ValueError("Unsupported modulation: " + selected_modulation)
    if selected_coding not in code_relative:
        raise ValueError("Unsupported coding rate: " + str(selected_coding))
    
    # Compute shift values based on k1 and k2.
    #max_shift_mod = 2**k1 - 1
    #max_shift_code = 2**k2 - 1
    max_shift_mod = k1 - 1
    max_shift_code = k2 - 1
    shift_mod = int(round(mod_relative[selected_modulation] * max_shift_mod))
    shift_code = int(round(code_relative[selected_coding] * max_shift_code))
    
    # Apply cyclic shifts.
    ctrl_mod_seq = np.roll(base_mod_seq, shift_mod)
    ctrl_code_seq = np.roll(base_code_seq, shift_code)
    
    # Concatenate to form control field.
    control_field = np.concatenate((ctrl_mod_seq, ctrl_code_seq))
    
    # Build full frequency-domain vector.
    freq_domain = np.zeros(fft_size, dtype=np.complex64)
    # Place sync sequence in the lower subcarriers.
    freq_domain[:sync_length] = zc_sync
    # Place control field in the upper subcarriers.
    freq_domain[sync_length:] = control_field
    
    # Convert frequency-domain vector to time domain.
    # Here we assume freq_domain is in centered order, so we apply ifftshift first.
    #time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain)).astype(np.complex64)
    time_domain = np.fft.ifft(freq_domain).astype(np.complex64)

    # Prepend cyclic prefix.
    cp = time_domain[-cp_len:]
    time_domain_symbol = np.concatenate((cp, time_domain))
    
    mcs_info = (selected_modulation, selected_coding, shift_mod, shift_code)
    return time_domain_symbol, freq_domain, control_field, mcs_info

def decode_sync_and_control_ofdm_symbol(received_symbol, fft_size, cp_len, k1, k2, u_sync=1, u_mod=1, u_code=1):
	candidate_mod_dict = {
						"QPSK":0.0,
						"16QAM":0.25,
						"64QAM":0.5,
						"256QAM":0.75}

	candidate_code_dict = {0.0833:0.000, 0.1666:0.166, 0.3333:0.333,
		0.5:    0.500,
		0.6667: 0.666,
		0.8333: 0.833
	}
						
	"""
	"BPSK":0.01, removed
	Decode an OFDM symbol (with CP) that carries a sync+control field.

	The transmitted frequency-domain vector was built as:
	   - Sync region (indices 0 to N_sync-1), with N_sync = fft_size - (k1+k2),
		 carrying a Zadoff–Chu sequence.
	   - Control region (indices N_sync to fft_size-1), where:
			• The first k1 subcarriers carry the modulation-control sequence,
			• The next k2 subcarriers carry the coding-control sequence.
	At the transmitter the control sequences were generated by taking a base Zadoff–Chu
	sequence (of length k1 for modulation, and k2 for coding) and cyclically shifting it by:
		 shift = round(relative * (2**k - 1))
	where relative is chosen from candidate dictionaries.

	This function performs the following:
	  1. Remove the CP and compute the FFT (via ifftshift then FFT).
	  2. Split the frequency-domain vector into sync region and control region.
	  3. From the control region, extract the modulation part (first k1 subcarriers)
		 and the coding part (next k2 subcarriers).
	  4. For each control part, test each candidate from candidate_mod_dict (for modulation)
		 and candidate_code_dict (for coding) by cyclically shifting the base Zadoff–Chu sequence
		 (of length k1 or k2) and computing the correlation with the received control portion.
		 The candidate with the highest correlation is chosen.
	  5. Return the estimated modulation type and coding rate.

	Parameters:
	  received_symbol : np.ndarray (complex64)
		  The received OFDM symbol (with CP) in time domain.
	  fft_size : int
		  Total FFT size.
	  cp_len : int
		  Length of cyclic prefix.
	  k1 : int
		  Number of subcarriers reserved for modulation control.
	  k2 : int
		  Number of subcarriers reserved for coding control.
	  candidate_mod_dict : dict
		  Mapping of modulation candidates to relative values (e.g., "QPSK":0.25).
	  candidate_code_dict : dict
		  Mapping of coding rate candidates to relative values (e.g., 0.5:0.03125).
	  u_sync, u_mod, u_code : int, optional
		  Zadoff–Chu roots for sync, modulation, and coding sequences.

	Returns:
	  results : dict with keys:
		 "sync_region":  the sync portion (frequency domain).
		 "estimated_modulation": estimated modulation string.
		 "estimated_coding": estimated coding rate.
		 "mod_shift": estimated shift value for modulation control.
		 "code_shift": estimated shift value for coding control.
	"""
	# Remove CP and compute FFT.
	time_symbol = received_symbol[cp_len:cp_len+fft_size]
	#freq_symbol = np.fft.fft(np.fft.ifftshift(time_symbol))
	#freq_symbol = np.fft.fftshift(np.fft.fft(time_symbol))
	print("Length/Size of time_symbol is ", len(time_symbol))
	freq_symbol = np.fft.fft(time_symbol).astype(np.complex64)

	#time_domain= np.fft.ifft(freq_full).astype(np.complex64)


	#time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain)).astype(np.complex64)

	# Determine allocation:
	total_control = k1 + k2
	N_sync = fft_size - total_control
	# Split frequency-domain vector:
	sync_region = freq_symbol[:N_sync]
	control_region = freq_symbol[N_sync:]

	# Split control region into modulation and coding parts.
	rec_mod = control_region[:k1]
	rec_code = control_region[k1:]

	# Local helper: generate base ZC sequence.
	def generate_zc_seq(length, u):
		n = np.arange(length)
		return np.exp(-1j * np.pi * u * n * (n+1) / length).astype(np.complex64)

	base_mod_seq = generate_zc_seq(k1, u_mod)
	base_code_seq = generate_zc_seq(k2, u_code)

	# Received modulation and coding bits are
	print(base_mod_seq)
	print(base_code_seq)

	print("Candidate search For modulation")
	# Candidate search for modulation control:
	max_shift_mod = k1 - 1
    #max_shift_mod = 2**k1 - 1
	best_corr_mod = -np.inf
	best_candidate_mod = None
	for candidate, rel in candidate_mod_dict.items():
		shift_candidate = int(round(rel * max_shift_mod))
		candidate_seq = np.roll(base_mod_seq, shift_candidate)
		corr = np.abs(np.vdot(rec_mod, candidate_seq))
		#corr = np.real(np.vdot(rec_mod, candidate_seq))
		print("Modulation Shift candidate and correlation are: ", shift_candidate, corr)
		if corr > best_corr_mod:
			#print (corr)
			best_corr_mod = corr
			best_candidate_mod = candidate
			best_mod_shift = shift_candidate

	print("Candidate search For coding")
		
	#if not (best_candidate_mod!="BPSK") and (best_candidate_cod not in nonBPSKNotAllowedCodeRate1)

	nonQPSKNotAllowedCodeRate1=[0.0833,0.1666,0.3333]
	# Candidate search for coding control:
	max_shift_code = k2 - 1
    #max_shift_code = 2**k2 - 1
	best_corr_code = -np.inf
	best_candidate_code = None
	for candidate, rel in candidate_code_dict.items():
		shift_candidate = int(round(rel * max_shift_code))
		candidate_seq = np.roll(base_code_seq, shift_candidate)
		corr = np.abs(np.vdot(rec_code, candidate_seq))
		#corr = np.real(np.vdot(rec_code, candidate_seq))
		print("Coding scheme shift candidate and correlation are: ", shift_candidate, corr)
		#print (corr)
		if corr > best_corr_code and best_candidate_mod=="QPSK":
			best_corr_code = corr
			best_candidate_code = candidate
			best_code_shift = shift_candidate
		elif corr > best_corr_code and (best_candidate_mod!="QPSK") and (candidate not in nonQPSKNotAllowedCodeRate1):
			best_corr_code = corr
			best_candidate_code = candidate
			best_code_shift = shift_candidate

	results = {
		"sync_region": sync_region,
		"estimated_modulation": best_candidate_mod,
		"estimated_coding": best_candidate_code,
		"mod_shift": best_mod_shift,
		"code_shift": best_code_shift
	}
	return results



def match_locs(detected, inserted, tolerance=2):
    correct = 0
    for loc in inserted:
        if any(abs(loc - d) <= tolerance for d in detected):
            correct += 1
    return correct


from numba import njit

@njit
def compute_normalized_corr_windows(rx_signal, ref_symbol, energy_ref, step, L, ref_len):
    num_steps = (L + step - 1) // step
    scores = np.zeros(num_steps, dtype=np.float32)
    for n in range(num_steps):
        i = n * step
        if i + ref_len > len(rx_signal):
            continue
        corr_real = 0.0
        corr_imag = 0.0
        energy = 0.0
        for j in range(ref_len):
            r = rx_signal[i + j]
            f = ref_symbol[j]
            energy += r.real**2 + r.imag**2
            corr_real += r.real * f.real + r.imag * f.imag
            corr_imag += r.imag * f.real - r.real * f.imag
        if energy > 0:
            corr_mag = np.sqrt(corr_real**2 + corr_imag**2)
            scores[n] = corr_mag / np.sqrt(energy * energy_ref)
    return scores



def energy_guided_top_corr2(rx_signal, ref_symbol, ref_len, energy_ref,
                           num_candidates=50, refine_half_win=75, top_k_final=12):
    L = len(rx_signal) - ref_len + 1
    
    '''
    energy_window = np.array([
        np.sum(np.abs(rx_signal[i:i + ref_len]*np.conjugate(ref_symbol)))
        for i in range(0, L, ref_len // 2)
    ])
    '''

    temp_start_t = time.time()

    energy_window = np.array([
    np.abs(np.vdot(ref_symbol, rx_signal[i:i + ref_len])) /
    np.sqrt(np.sum(np.abs(rx_signal[i:i + ref_len])**2) * energy_ref)
    for i in range(0, L, ref_len // 2)
    ])

    temp_end_t = time.time()
    print("Current execution  time (with np.dot) = ", temp_end_t-temp_start_t)


    '''

    temp_start_t = time.time()

    step = ref_len // 2
    energy_window = compute_normalized_corr_windows(rx_signal, ref_symbol, energy_ref, step, L, ref_len)

    temp_end_t = time.time()
    print("Current executio  time (with numba njit) = ", temp_end_t-temp_start_t)

    '''
    candidate_offsets = np.argsort(energy_window)[-num_candidates:]
    candidate_indices = (candidate_offsets * (ref_len // 2)).clip(max=L - 1)

    # Refine around energy peaks
    refined_peaks = []
    seen = set()
    for coarse_idx in candidate_indices:
        start = max(0, coarse_idx - refine_half_win)
        end = min(L, coarse_idx + refine_half_win)
        best_val = -1
        best_idx = -1
        for i in range(start, end):
            if i in seen: continue  # avoid duplicate effort
            seen.add(i)
            candidate = rx_signal[i: i + ref_len]
            energy_candidate = np.sum(np.abs(candidate)**2)
            if energy_candidate != 0:
                corr_val = np.sum(candidate * np.conjugate(ref_symbol))
                val = np.abs(corr_val) / np.sqrt(energy_candidate * energy_ref)
                if val > best_val:
                    best_val = val
                    best_idx = i
        if best_idx >= 0:
            refined_peaks.append((best_idx, best_val))

    # Sort by score and return top unique indices
    refined_peaks = list(set(refined_peaks))  # remove duplicates
    refined_peaks.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in refined_peaks[:top_k_final]]
    return sorted(top_indices)



def detect_sync_start(R, period=1152, num_repeats=10):
    """
    Detect the SYNC start by computing periodic correlation energy sum.

    Args:
        R (np.ndarray): Normalized correlation magnitude array (1D)
        period (int): Period between SYNC symbols (default: 1152 samples)
        num_repeats (int): How many periods to accumulate over (default: 10)

    Returns:
        int: Index of the best SYNC start
    """
    R = np.abs(R)  # Ensure we're working with magnitude
    R = R[:period*num_repeats]
    N = len(R)

    '''
    plt.figure()
    plt.plot(R)
    plt.xlabel("Start Indexes")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.show()
    '''

    # Truncate R to ensure reshaping works without overflow
    max_start = N - period * (num_repeats - 1)
    #R = R[:max_start]

    print("Lengths of R, period and num-repeats (R=Period*num-repeats) = ", len(R), period,num_repeats)

    # Stack num_repeats rows, each offset by one period
    idx = np.arange(num_repeats)[:, None] * period + np.arange(max_start)
    valid_mask = (idx[-1] < N)  # Only keep columns where the last index is valid
    idx = idx[:, valid_mask]

    # Gather values and sum across rows (i.e., accumulate across time)
    scores = R[idx].sum(axis=0)
    
    '''
    plt.figure()
    plt.plot(scores)
    plt.xlabel("Start Indexes")
    plt.ylabel("Magnitude of start indexes")
    plt.grid(True)
    plt.show()
    '''

    # Get index with max score → first SYNC start
    best_index = np.argmax(scores)

    return idx[0, best_index]

import heapq

def synchronize_repetitive_ofdm_blocks(
    rx_signal: np.ndarray,
    sync_freq_subcarriers: np.ndarray,  # Zadoff–Chu for sync subcarriers; length = fft_size - (k1+k2)
    fft_size: int,
    cp_len: int,
    data_block_len: int,
    K: int,
    G: int,
    sampling_rate: float,
    search_length: int = None,
    spacing_tolerance: int = None
):
    """
    Searches for candidate start indices of the sync+control OFDM symbol in rx_signal.
    
    Transmitter standard:
      - Frequency-domain vector of length fft_size.
      - The first sync_size subcarriers (sync_size = fft_size - (k1+k2)) are filled with a Zadoff–Chu sequence.
      - The remaining (k1+k2) subcarriers (control info) are unknown at the receiver.
      - The time–domain symbol is generated by a plain ifft (no shift) and a cyclic prefix (length cp_len) is prepended.
    
    Receiver processing:
      For each candidate offset (from 0 to search_length – (cp_len+fft_size)):
        1. Extract a candidate block of length sc_block_len = cp_len + fft_size.
        2. Compute the normalized correlation between the candidate block and a reference symbol (constructed exactly as at the transmitter).
        3. Only candidates with normalized correlation above 80% of the maximum are retained.
        4. A local refinement (±3 samples) is applied to each candidate.
        5. CFO is estimated using CP correlation:
             CFO (Hz) = (angle(sum(received_CP * conj(tail))) * sampling_rate) / (2π * cp_len)
           where received_CP is the first cp_len samples and tail is the last cp_len samples of the candidate block.
    
    Inputs remain unchanged.
    
    Returns a dictionary with keys:
       "candidate_starts": list of candidate start indices (integers)
       "candidate_cfos"  : list of CFO estimates (floats, in Hz) for each candidate
       "block_size"      : cp_len + fft_size + data_block_len (expected block length in samples)
    """
    # Derived lengths
    sc_block_len = cp_len + fft_size          # Length of one sync+control OFDM symbol (with CP)
    block_size = sc_block_len + data_block_len  # Overall block length
    
    if search_length is None or search_length > len(rx_signal):
        search_length = len(rx_signal)
    if search_length < sc_block_len:
        raise ValueError("search_length is less than the length of one sync+control symbol.")
    if spacing_tolerance is None:
        spacing_tolerance = block_size // 2
    
    # sync_size as defined by transmitter.
    sync_size = len(sync_freq_subcarriers)
    if sync_size > fft_size:
        raise ValueError("Length of sync_freq_subcarriers must be <= fft_size.")
    
    # Build the reference symbol exactly as at the transmitter.
    freq_ref = np.zeros(fft_size, dtype=np.complex64)
    freq_ref[:sync_size] = sync_freq_subcarriers
    time_ref = np.fft.ifft(freq_ref).astype(np.complex64)  # plain ifft
    cp_ref = time_ref[-cp_len:]
    ref_symbol = np.concatenate([0*cp_ref, time_ref])  # Length = cp_len + fft_size (I multiply the cp part by 0 since not needed)
    #ref_symbol = time_ref # I think this should be the correct one (i.e., multiplying cp by 0)!
    
    # Pre-compute the energy of the reference symbol.
    energy_ref = np.sum(np.abs(ref_symbol)**2)

    adaptive_sync_time = time.time()
    # Compute normalized cross-correlation by sliding the reference over rx_signal.
    L = search_length - sc_block_len + 1
    norm_corr = np.zeros(L, dtype=np.float32)
    startScale1=0.45
    endScale1=0.55
    for i in range(int(startScale1*L),int(endScale1*L)):
        candidate = rx_signal[i: i + sc_block_len]
        energy_candidate = np.sum(np.abs(candidate)**2)
        if energy_candidate == 0:
            norm_corr[i] = 0
            #print("yes")
        else:
            corr_val = np.sum(candidate * np.conjugate(ref_symbol))
            norm_corr[i] = np.abs(corr_val) / np.sqrt(energy_candidate * energy_ref)
    
    #print(L,i)
    #print("Time to correlate (in sec) (This needs some thinking to improve)", time.time() - start_fast)

    norm_corr=norm_corr[int(startScale1*L):int(endScale1*L)]
    # R is your 1D normalized correlation array (real-valued)
    sync_start = detect_sync_start(norm_corr, period=block_size, num_repeats=len(norm_corr)//block_size)
    print("Best SYNC start found at index:", sync_start)

    candidate_starts0 = []
    temp_energy0 = []
    initialStart1=int(startScale1*L) + sync_start
    for i in range (-initialStart1//block_size, (L-initialStart1)//block_size-1):
        tempStart1=int(startScale1*L) + sync_start + i*block_size
        candidate_starts0.append(tempStart1)
        candidate = rx_signal[tempStart1: tempStart1 + sc_block_len]
        energy_candidate = np.sum(np.abs(candidate)**2)
        if energy_candidate == 0:
            temp_energy0.append(0)
        else:
            corr_val = np.sum(candidate * np.conjugate(ref_symbol))
            temp_energy0.append(np.abs(corr_val) / np.sqrt(energy_candidate * energy_ref))

    energy_theshold=heapq.nlargest(int(0.1*len(temp_energy0)), temp_energy0)[-1]
    print ("All Energy values", temp_energy0)
    print ("Energy Threshold", energy_theshold)
    candidate_starts = []
    candidate_energies = []
    if energy_theshold>0:
        for i in range(len(temp_energy0)):
            if temp_energy0[i]>0.8*energy_theshold:
                candidate_starts.append(candidate_starts0[i])
                candidate_energies.append(temp_energy0[i])

    #candidate_starts=candidate_starts[int(0.4*len(candidate_starts)):int(0.6*len(candidate_starts))]
    #time.sleep(20)

    #candidate_starts=candidate_starts[:100]

    time_adaptive = time.time() - adaptive_sync_time
    #print(f"Detected Indices: {sorted(candidate_starts)}\n")
    print(f"Execution Time (Adaptive method): {time_adaptive:.2f} s")
    print("Number of detected sync positions:", len(candidate_starts))

    time.sleep(10)

    '''
    # Use threshold = 80% of maximum normalized correlation.
    threshold = 0.8 * np.max(norm_corr)
    valid_idx = np.where(norm_corr >= threshold)[0]
    if valid_idx.size == 0:
        valid_idx = np.array([np.argmax(norm_corr)])
    valid_idx.sort()  # sort in ascending order
    
    # Enforce spacing: select up to G candidates that are at least spacing_tolerance apart.
    candidate_starts = []
    for idx in valid_idx:
        if len(candidate_starts) >= G:
            break
        if all(abs(idx - s) >= spacing_tolerance for s in candidate_starts):
            # Local refinement: search within ±3 samples to maximize normalized correlation.
            refine_window = 3
            left = max(0, idx - refine_window)
            right = min(L - 1, idx + refine_window)
            window_indices = np.arange(left, right + 1)
            refined_idx = int(window_indices[np.argmax(norm_corr[left:right + 1])])
            candidate_starts.append(refined_idx)

    candidate_starts_true=candidate_starts
    time_taken = time.time() - start_fast
    print(f"Execution Time (Brute force method): {time_taken:.2f} s")
    print(f"Accurate Detected Indices: {sorted(candidate_starts_true)}\n")

    start_fast = time.time()
    ref_len=len(ref_symbol)
    candidate_starts = energy_guided_top_corr2(rx_signal, ref_symbol, ref_len, energy_ref,
                                       num_candidates=int(5*G), refine_half_win=int(0.25*ref_len), top_k_final=int(1.25*G))
    #correct_fast = match_locs(top_locs_fast, insert_locs, tolerance=2)
    #candidate_starts= match_locs(top_locs_fast, insert_locs, tolerance=2)

    time_fast = time.time() - start_fast
    print(f"Execution Time (Fast Approximation method): {time_fast:.2f} s")
    print(f"Detected Indices: {sorted(candidate_starts)}\n")
    print("Number of detected sync positions:", len(candidate_starts))

    time.sleep(10)

    '''
    #candidate_starts, candidate_starts_true
    #correct_fast = match_locs(candidate_starts, candidate_starts_true, tolerance=0)
    #print(f"Correct Detections: {correct_fast}/{G}")

    #print("See the difference in execution time and candidate starts!")
    # It is already shited where center = 0
    #plt.figure()
    #plt.plot(np.abs(ref_symbol))
    #plt.grid(True)
    #plt.show()

    #print (ffr)
    # CFO estimation: use CP correlation method.
        
    def estimate_cfo_cp(candidate_block, cp_len, sampling_rate):
        """
        Estimate CFO from a candidate OFDM symbol block using CP correlation.
        
        Assumes candidate_block has length = cp_len + fft_size, where the first cp_len samples are the CP,
        and the last cp_len samples of the block (the tail) are an exact copy of the CP in the transmitter.
        
        Under a CFO f, the phase difference between the CP and the tail is:
            phase_diff = -2π * f * fft_size / sampling_rate.
        
        This function computes:
            f_est = - (phase_diff/(2π)) * (sampling_rate / fft_size)
        
        Parameters:
        candidate_block : np.ndarray
            Time-domain candidate block (with CP) from the receiver.
        cp_len : int
            Length of the cyclic prefix.
        sampling_rate : float
            Sampling rate in Hz.
        
        Returns:
        f_est : float
            Estimated CFO in Hz.
        """
        #print(len(candidate_block))
        fft_size = len(candidate_block) - cp_len  # derive FFT size from candidate block length
        cp_received = candidate_block[:cp_len]
        tail = candidate_block[-cp_len:]
        product = np.sum(cp_received * np.conjugate(tail))
        phase_diff = np.angle(product)
        f_est = - phase_diff / (2 * np.pi) * (sampling_rate / fft_size)
        return f_est


    rxToEstimateNoisePower1=[]
    #energy1=[]
    #syncPower1=np.vdot(sync_freq_subcarriers,sync_freq_subcarriers)
    #freq_ref[:sync_size] = sync_freq_subcarriers
    candidate_cfos = []
    for start in candidate_starts:
        if start + sc_block_len <= len(rx_signal):
            block = rx_signal[start: start + sc_block_len]
            cfo_est = estimate_cfo_cp(block, cp_len, sampling_rate)
            candidate_cfos.append(cfo_est)
            #tempRx1=1/(np.sqrt(fft_size))*np.fft.fft(rx_signal[start+cp_len: start + cp_len+fft_size]).astype(np.complex64)
            tempRx1=np.fft.fft(rx_signal[start+cp_len: start + cp_len+fft_size]).astype(np.complex64)
            tempRx1=tempRx1[:len(sync_freq_subcarriers)]
            rxToEstimateNoisePower1.append(tempRx1)
            #tempRx1=tempRx1[:len(sync_freq_subcarriers)]
            #energy1.append(np.vdot(tempRx1, tempRx1))
        else:
            candidate_cfos.append(None)

    #aveSyncPlusNoisePower1=sum(energy1)/len(energy1)
    
    return {
        "candidate_starts": candidate_starts,
        "candidate_cfos": candidate_cfos,
        "block_size": block_size,
        "sync_and_rx_signal_for_noise_est": [sync_freq_subcarriers,rxToEstimateNoisePower1]
    }

def correct_frequency_offset(rx_signal: np.ndarray, freq_offset_hz: float, sampling_rate: float, start_index: int = 0) -> np.ndarray:
    """
    Correct the frequency offset in rx_signal by multiplying by a complex exponential.
    The phase correction is referenced to start_index.
    """
    n = np.arange(len(rx_signal))
    correction = np.exp(-1j * 2 * np.pi * freq_offset_hz * (n - start_index) / sampling_rate)
    return rx_signal * correction


import shutil

def delete_folder_if_exists(folder_path):
    """Delete the folder at folder_path if it exists."""
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def commonTxRxParameters(FFT_SIZE):

    if FFT_SIZE==256:
        CP_LEN=32
    elif FFT_SIZE==512:
        CP_LEN=64
    elif FFT_SIZE==1024:
        CP_LEN=128
    elif FFT_SIZE==2048:
        CP_LEN=256
    elif FFT_SIZE==4096:
        CP_LEN=512

    TT = 5.0         # total trasnsmission time 
    KK = 100          # repeat one symbol block KK times => pattern
    #RU_SIZE = 26
    YY = 5  # This shows the number of file names I should check (This can be higher)
    GG = 90
    # How many candidate timings I should pick from a single file saving (This must be SNR dependent with lookup table)
    #FFT_SIZE= 512
    CP_LEN = 32
    subcarrier_spacing = 15.0e3
    tx_sampling_rate = FFT_SIZE* subcarrier_spacing
    rx_sampling_rate = FFT_SIZE* subcarrier_spacing
    sampling_rate = FFT_SIZE * subcarrier_spacing  # 20e6 Hz
    #TX_FREQ = 2.437e9
    #RX_FREQ = 2.437e9
    TX_FREQ = 2.637e9
    RX_FREQ = 2.437e9
    TX_GAIN = 70
    RX_GAIN = 60

    SYNC_DELAY = 0.1

    # data_mod_user= "16QAM", data_rate_user= 0.75
    #modulation= "16QAM"
    #code_rate= 0.75

    pilot_positions=pilotSubCarrierIndexes1(FFT_SIZE)
    [usedSCmin1,usedSCmax1]=usedSubCarrierIndexes1(FFT_SIZE)

    #pilot_positions= [5,20]
    pilot_value= 1+1j

    pilotPosWithInRSCMinSCMax1=[]
    for i in range(usedSCmin1,usedSCmax1):
        if i in pilot_positions:
            pilotPosWithInRSCMinSCMax1.append(i)

    max_used_subcarriers_excluding_pilot=usedSCmax1-usedSCmin1-len(pilotPosWithInRSCMinSCMax1)
    input_crc_size=16

    k1 = int (FFT_SIZE/32)
    k2 = int (FFT_SIZE/32)

    #k1=9
    #k2=9

    A1=[]
    A1.append(TT)
    A1.append(KK)
    A1.append(YY)
    A1.append(GG)
    A1.append(k1)
    A1.append(k2)
    A1.append(subcarrier_spacing)
    A1.append(tx_sampling_rate)
    A1.append(rx_sampling_rate)
    A1.append(TX_FREQ)
    A1.append(RX_FREQ)
    A1.append(SYNC_DELAY)
    A1.append(pilot_positions)
    A1.append(pilot_value)
    A1.append(usedSCmin1)
    A1.append(usedSCmax1)
    A1.append(pilotPosWithInRSCMinSCMax1)
    A1.append(input_crc_size)
    A1.append(max_used_subcarriers_excluding_pilot)
    A1.append(CP_LEN)
    A1.append(TX_GAIN)
    A1.append(RX_GAIN)

    return A1


def compute_size_of_Tx_Rx_bits_and_subcarriers(FFT_SIZE,modulation,desired_rate,input_crc_size):

    b_dict={'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
    if modulation not in b_dict:
        raise ValueError("Mod not supported!")
    b=b_dict[modulation]

    base_mat = None
    for r,mat in TOY_80211AX_MATRICES.items():
        if abs(r - desired_rate)<1e-3:
            base_mat= mat
            break
    if base_mat is None:
        raise ValueError(f"No toy base matrix found for code_rate={desired_rate}.")

    #global SMALL_R12_BASE
    #base_mat = np.array(SMALL_R12_BASE, dtype=int)  # shape (4,8)
    #base_mat= np.array(base_mat, dtype=int)
    m, n= base_mat.shape  # e.g. (12,24) for rate=1/2

    num_subcarriers=FFT_SIZE
    # total coded bits T
    T=num_subcarriers*b
    # We do floor division for z:
    z=T//n
    # final codeword length
    N=n*z
    # If N not multiple of b, we further reduce z so that N is multiple of b
    while (N % b)!=0 or (N % input_crc_size)!=0:
        z-=1
        if z<=0:
            raise ValueError("Cannot find z>0 such that N is multiple of b.")
        N=n*z
    # Now we have N <= T and N multiple of b.
    # subcarriers used
    used_subcarriers=N//b
    unused_subcarriers=(T//b)-used_subcarriers

    print("Used sub-carriers=", used_subcarriers)
    print("Unused sub-carriers=", unused_subcarriers)
    # info bits
    K=(n-m)*z
    return [K,used_subcarriers,unused_subcarriers]


def getDataSubcarrierIndexes1(FFT_SIZE,mod_sig_size,pilot_positions):
    dataSubCarrierIndexes1=[]

    FFT_SIZE-mod_sig_size-len(pilot_positions)

    # Create the full frequency-domain vector
    #freq_full = np.zeros(FFT_SIZE, dtype=np.complex64)
    ct1=-1
    # I leave some sub-carriers left unused (Just the DC)
    unUsed1=5
    desRange1=list(range(unUsed1,int(FFT_SIZE/2)-1))
    #desRange1=desRange1 + (list(range(int(FFT_SIZE/2)+1,FFT_SIZE-1)))
    #desRange1=range(unUsed1,int(FFT_SIZE/2)-1) + range(int(FFT_SIZE/2)+1,FFT_SIZE-1)
    unDesiredSC1=range(int(FFT_SIZE/2-(FFT_SIZE-len(pilot_positions)-mod_sig_size)/2)+unUsed1,int(FFT_SIZE/2+(FFT_SIZE-len(pilot_positions)-mod_sig_size)/2))
    print(min)
    for i in range(unUsed1,FFT_SIZE-1):
        if i not in pilot_positions and ct1 < mod_sig_size-1 and i not in unDesiredSC1:
            ct1=ct1+1
            dataSubCarrierIndexes1.append(i)
    dataSubCarrierIndexes1=sorted(dataSubCarrierIndexes1)
    print("Total used sub-carrier=", len(dataSubCarrierIndexes1))
    print("Total size of modulated signals=", mod_sig_size)
    print("Data sub-carier indexes", dataSubCarrierIndexes1[:10])
    print("Pilot sub-carier indexes", pilot_positions)

    return dataSubCarrierIndexes1



def normalize_and_clip(time_domain_signal, x_percent):
    """
    Normalize the time-domain signal using the x-th percentile of the
    magnitudes as the desired peak, then clip samples with normalized
    magnitude > 1 so that they have unit magnitude.

    Parameters:
    -----------
    time_domain_signal : numpy array (complex)
        The input time-domain signal.
    x_percent : float
        The percentile (e.g., 95 for the 95th percentile) used to
        define the desired peak value for normalization.

    Returns:
    --------
    clipped_signal : numpy array (complex)
        The normalized and intentionally clipped signal.
    """
    # 1. Compute the desired peak value from the x-th percentile.
    des_peak_val = np.percentile(np.abs(time_domain_signal), x_percent)
    
    # 2. Normalize the entire signal by this desired peak value.
    normalized_signal = time_domain_signal / des_peak_val
    
    # 3. For samples with magnitude > 1, clip them to unit magnitude.
    #    We do this elementwise, preserving the phase.
    clipped_signal = np.copy(normalized_signal)
    indices = np.abs(clipped_signal) > 1
    clipped_signal[indices] = clipped_signal[indices] / np.abs(clipped_signal[indices])
    
    return clipped_signal


def compute_snr_db(tx_power_mW, distance_m, carrier_freq_Hz, bandwidth_Hz,
                   tx_gain_dB, rx_gain_dB):
    """
    Computes the SNR (in dB) under a simple free-space path-loss model, given:

      - tx_power_mW   : Transmit power in milliwatts
      - distance_m    : Distance between Tx and Rx in meters
      - carrier_freq_Hz : Carrier frequency in Hz
      - bandwidth_Hz  : System bandwidth in Hz
      - tx_gain_dB    : Transmitter gain in dB
      - rx_gain_dB    : Receiver gain in dB

    Assumptions:
      - Free-space path loss (FSPL)
      - Room temperature thermal noise: -174 dBm/Hz
      - No additional losses (other than FSPL)

    Returns:
      snr_dB : Estimated SNR in dB
    """
    
    # 1) Convert transmit power from mW to dBm
    tx_power_dBm = 10.0 * np.log10(tx_power_mW)
    
    # 2) Calculate free-space path loss (FSPL) in dB
    #    distance in km, frequency in MHz
    distance_km = distance_m / 1000.0
    freq_MHz = carrier_freq_Hz / 1e6

    # FSPL(dB) = 20 * log10(distance_km) + 20 * log10(freq_MHz) + 32.44
    fspl_dB = 20.0 * np.log10(distance_km) + 20.0 * np.log10(freq_MHz) + 32.44
    
    # 3) Received power in dBm:
    #    rx_power_dBm = tx_power_dBm + tx_gain_dB - fspl_dB + rx_gain_dB
    rx_power_dBm = tx_power_dBm + tx_gain_dB - fspl_dB + rx_gain_dB
    
    # 4) Thermal noise in dBm over bandwidth
    #    noise_floor_dBm = -174 dBm/Hz + 10*log10(BW in Hz)
    noise_floor_dBm = -174.0 + 10.0 * np.log10(bandwidth_Hz)
    
    # 5) SNR in dB
    snr_dB = rx_power_dBm - noise_floor_dBm
    return snr_dB


import numpy as np

def estimate_tx_power_mW(time_domain_signal,
                         NFFT,
                         subcarrier_spacing_hz,
                         carrier_freq_hz):
    """
    Estimate the transmit power (in milliwatts) of an OFDM signal
    on a B200 mini USRP at ~0 dB TX gain, incorporating a rough
    frequency-dependent maximum power model.

    time_domain_signal     : numpy array (complex),
                             The time-domain OFDM samples (baseband)
    NFFT                   : int,
                             Number of subcarriers in the OFDM IFFT
    subcarrier_spacing_hz  : float,
                             Sub-carrier spacing in Hz
    carrier_freq_hz        : float,
                             RF carrier frequency in Hz

    Returns:
      tx_power_mW : float
        Estimated transmit power in milliwatts
    """

    # --------------------------------------------------------------
    # 1) (Optional) Compute sampling rate (for reference only)
    sampling_rate = NFFT * subcarrier_spacing_hz

    # --------------------------------------------------------------
    # 2) Compute average digital power of the time-domain signal
    avg_digital_power = np.mean(np.abs(time_domain_signal)**2)
    if avg_digital_power <= 0:
        return 0.0

    # dB full-scale (dBFS) if 1.0 = "full-scale" average
    avg_digital_power_dBFS = 10.0 * np.log10(avg_digital_power)

    print("Average digital power in linear scale = ", avg_digital_power)

    print("Average digital power in full scale (in dB) = ", avg_digital_power_dBFS)

    # --------------------------------------------------------------
    # 3) Approximate frequency-dependent max output power for B200 mini
    #    (Very rough piecewise model. Adjust or expand as needed.)

    # Default to +10 dBm (10 mW) as a mid-band guess
    # Then reduce it at the "extremes" or handle known frequency bands:
    if 1.0e9 <= carrier_freq_hz <= 3.0e9:
        max_power_dBm = 10.0   # ~+10 dBm in mid-band
    else:
        max_power_dBm = 8.0    # ~+8 dBm outside that range (example)

    # --------------------------------------------------------------
    # 4) Map digital power to actual output power
    #
    # If avg_digital_power_dBFS = 0, that means the signal is "full-scale",
    # which we approximate => max_power_dBm at that freq.
    #
    # If the signal is below full-scale, we reduce accordingly:
    #
    #   Tx power (dBm) = max_power_dBm + avg_digital_power_dBFS
    #
    tx_power_dBm = max_power_dBm + avg_digital_power_dBFS

    # --------------------------------------------------------------
    # 5) Convert from dBm to mW
    tx_power_mW = 10.0 ** (tx_power_dBm / 10.0)

    return tx_power_mW


def generate_qpsk_symbols(NFFT):
    """
    Generate a random QPSK symbol sequence for the frequency domain.
    
    Parameters:
    -----------
    NFFT : int
        Number of QPSK symbols (subcarriers).
        
    Returns:
    --------
    freq_domain_data : numpy array (complex)
        Random QPSK symbols normalized by sqrt(2).
    """
    # Define QPSK constellation: normalized by sqrt(2)
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    # Generate random indices for each subcarrier
    indices = np.random.randint(0, 4, NFFT)
    
    # Map indices to QPSK symbols
    freq_domain_data = constellation[indices]
    return freq_domain_data



import numpy as np

def iterative_clipping_and_filtering(freq_domain_data, clipping_threshold, num_iterations, mask=None):
    """
    Perform Iterative Clipping and Filtering (ICF) on an OFDM signal.
    
    The process is:
      1. Compute the time-domain signal via IFFT.
      2. Clip the time-domain signal such that any sample with magnitude greater than
         the clipping_threshold is scaled back to that threshold (preserving phase).
      3. Convert back to frequency domain via FFT.
      4. Apply a frequency-domain mask to force out-of-band subcarriers (if provided).
      5. Repeat steps 1-4 for the specified number of iterations.
      
    Parameters:
    -----------
    freq_domain_data : np.ndarray (complex)
        Frequency-domain OFDM symbol (e.g., array of size NFFT).
    clipping_threshold : float
        Maximum allowed amplitude in time domain. Samples with |x[n]| > threshold are clipped.
    num_iterations : int
        Number of ICF iterations.
    mask : np.ndarray, optional
        Frequency-domain mask of shape (NFFT,). Values should be 1 for in‑band subcarriers and 0 for out‑of‑band.
        If None, a mask of ones (i.e. full band used) is assumed.
    
    Returns:
    --------
    low_papr_time_signal : np.ndarray (complex)
        Time‑domain OFDM signal after iterative clipping and filtering.
    """
    # Ensure freq_domain_data is a numpy array
    freq_domain_data = np.asarray(freq_domain_data)
    NFFT = freq_domain_data.size

    # If no mask is provided, assume full band (all ones)
    if mask is None:
        mask = np.ones(NFFT)
    else:
        mask = np.asarray(mask)
        if mask.size != NFFT:
            raise ValueError("Mask must have the same size as freq_domain_data.")

    # Initial IFFT to obtain the time-domain signal
    time_signal = np.fft.ifft(freq_domain_data)
    
    for _ in range(num_iterations):
        # --- Clipping in the time domain ---
        magnitudes = np.abs(time_signal)
        # Find samples exceeding the clipping threshold
        clip_idx = magnitudes > clipping_threshold
        # Clip these samples while preserving phase:
        time_signal[clip_idx] = (time_signal[clip_idx] / magnitudes[clip_idx]) * clipping_threshold
        
        # --- Convert back to frequency domain ---
        freq_temp = np.fft.fft(time_signal)
        
        # --- Filtering: Apply the mask to retain the original in-band frequency components ---
        # Here we force the out-of-band subcarriers to zero.
        freq_temp = freq_temp * mask
        
        # --- Convert back to time domain for the next iteration ---
        time_signal = np.fft.ifft(freq_temp)
    

    preserve_max=True
    # Optionally, renormalize so that the maximum amplitude is 1.
    if preserve_max:
        max_val = np.max(np.abs(time_signal))
        if max_val > 0:
            time_signal = time_signal / max_val

    # The output is the time-domain signal with reduced PAPR.
    low_papr_time_signal = time_signal
    return low_papr_time_signal


def main():
    """
    Single-user main with code_rate=0.75, modulation=16QAM, RU_size=26 subcarriers.
    We'll do a single RU in subcarriers [0..25], insert pilot at positions [5,20],
    everything else is data. Then we place that RU in freq-domain bins [0..25],
    the rest remain zero. We do IFFT+CP, replicate KK times => tile to TT seconds,
    then either SIMULATE or do actual continuous TX and RX.
    """
    SIMULATE = True  # Set True => no hardware, just AWGN loopback

    FFT_SIZE=512

    A1=commonTxRxParameters(FFT_SIZE)

    TT=A1[0]
    KK=A1[1]
    YY=A1[2]
    GG=A1[3]
    k1=A1[4]
    k2=A1[5]
    subcarrier_spacing=A1[6]
    tx_sampling_rate=A1[7]
    rx_sampling_rate=A1[8]
    TX_FREQ=A1[9]
    RX_FREQ=A1[10]
    SYNC_DELAY=A1[11]
    pilot_positions=A1[12]
    pilot_value=A1[13]
    usedSCmin1=A1[14]
    usedSCmax1=A1[15]
    pilotPosWithInRSCMinSCMax1=A1[16]
    input_crc_size=A1[17]
    max_used_subcarriers_excluding_pilot=A1[18]
    CP_LEN=A1[19]
    TX_GAIN=A1[20]
    RX_GAIN=A1[20]
    #A1.append(RX_GAIN)

    #subcarrier_spacing_hz = 15000.0   # e.g., 15 kHz
    subcarrier_spacing_hz = subcarrier_spacing
    #carrier_freq_hz = 2.4e9          # 2.4 GHz center frequency
    carrier_freq_hz = RX_FREQ

    #sampling_rate=NFFT*subcarrier_spacing_hz

    sampling_rate=FFT_SIZE*subcarrier_spacing

    # Generate random frequency-domain data
    np.random.seed(0)
    #freq_domain_data = (np.random.randn(NFFT) + 1j*np.random.randn(NFFT)) / np.sqrt(2)

    # OFDM parameters
    NFFT = FFT_SIZE

    freq_domain_data = generate_qpsk_symbols(NFFT)

    avg_digital_power_Freq_Domain = np.mean(np.abs(freq_domain_data)**2)
    print("Average Frequency domain power = ",avg_digital_power_Freq_Domain )

    time_domain_signal = np.fft.ifft(freq_domain_data) * np.sqrt(NFFT)
    avg_digital_power_Time_Domain = np.mean(np.abs(time_domain_signal)**2)
    print("Average Time domain power = ",avg_digital_power_Time_Domain )

    '''
    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(time_domain_signal))
    plt.title("Magnitude of freq_full (DC at index 0)")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()
	'''
	
    # Normalize so peak amplitude is near 1
    peak_val = np.max(np.abs(time_domain_signal))
    time_domain_signal /= peak_val

    #x_percent=98 (Simple clipping works bad)
    #clipped_signal = normalize_and_clip(time_domain_signal, x_percent)

    # Parameters for ICF
    clipping_threshold = 0.8      # For example, clip time-domain amplitudes above 0.8
    num_iterations = 25            # Number of iterations to perform
    
    # (Optional) Define a frequency mask (here, assume all subcarriers are used)
    mask = np.ones(NFFT)    
    # Run ICF
    low_papr_time_signal = iterative_clipping_and_filtering(freq_domain_data,clipping_threshold,num_iterations,mask)

    #clipped_signal=low_papr_time_signal
    #clipped_signal=time_domain_signal
    
    # Print some statistics
    #print("Desired peak value (x-th percentile):", np.percentile(np.abs(time_domain_signal), x_percent))
    #print("Max magnitude after normalization and clipping:", np.max(np.abs(clipped_signal)))

    # Estimate TX power
    tx_power_mW_est = estimate_tx_power_mW(time_domain_signal,
                                           NFFT,
                                           subcarrier_spacing_hz,
                                           carrier_freq_hz)

    '''
    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(time_domain_signal))
    plt.title("High PARP")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()
	'''

    print(f"Estimated TX power @ 2.4 GHz (B200mini @0dB gain): Original signal: {tx_power_mW_est:.3f} mW")

    # Estimate TX power
    tx_power_mW_est_low_parp = estimate_tx_power_mW(low_papr_time_signal,
                                           NFFT,
                                           subcarrier_spacing_hz,
                                           carrier_freq_hz)
    print(f"Estimated TX power @ 2.4 GHz (B200mini @0dB gain): Low PARP signal: {tx_power_mW_est_low_parp:.3f} mW")

	
    # It is already shited where center = 0
    #plt.figure()
    #plt.plot(np.abs(low_papr_time_signal))
    #plt.title("Low PARP")
    #plt.xlabel("Subcarrier Index")
    #plt.ylabel("Magnitude")
    #plt.grid(True)
    #plt.show()
	
    tx_power_mW     = tx_power_mW_est         # 100 mW
    distance_m      = 1000.0        # 1 km
    carrier_freq_Hz = carrier_freq_hz         # 2.4 GHz
    bandwidth_Hz    = sampling_rate           # 1 MHz
    
    if SIMULATE:
        tx_gain_dB      = 0           # 2 dB Tx gain
        rx_gain_dB      = 0           # 3 dB Rx gain
    else:
        tx_gain_dB      = 0           # 2 dB Tx gain
        rx_gain_dB      = 0           # 3 dB Rx gain


    snr_db_value = compute_snr_db(tx_power_mW, distance_m, carrier_freq_Hz, 
                                  bandwidth_Hz, tx_gain_dB, rx_gain_dB)
    
    print(f"SNR in dB (without parp reduction): {snr_db_value:.2f} dB")

    snr_db_value_low_parp = compute_snr_db(tx_power_mW_est_low_parp, distance_m, carrier_freq_Hz, 
                                  bandwidth_Hz, tx_gain_dB, rx_gain_dB)
    print(f"SNR in dB (with parp reduction): {snr_db_value_low_parp:.2f} dB")

    # Local system parameters (only these inputs):
    SNR_dB = snr_db_value_low_parp   # SNR in dB

    #print(ddf)
    #possible_code_rates=[0.5, 0.6667, 0.75, 0.8333] # 0.7s is not working

    MCS1 = lookUpTableSubCarrierMCS(SNR_dB,max_used_subcarriers_excluding_pilot)
    print(MCS1)

    modulation=MCS1[0]
    code_rate=MCS1[1]

    print("Tx Modulation=", modulation)
    print("Tx Code Rate=", code_rate)

    # Choose modulation and coding from the supported set.
    symbol, freq_vec, ctrl_field, mcs_info = generate_sync_and_control_ofdm_symbol(FFT_SIZE, CP_LEN, k1, k2, modulation, 
                                                                                   code_rate, u_sync=1, u_mod=1, u_code=1)

    print("Freq vector = ", len(freq_vec))
    #low_papr_time_signal = iterative_clipping_and_filtering(freq_vec,clipping_threshold,num_iterations,mask)
    #symbol=low_papr_time_signal

    print("Max used subcarriers excluding pilot=", max_used_subcarriers_excluding_pilot)
    print("OFDM symbol (with CP) length =", len(symbol))
    print("Control field (length =", len(ctrl_field), "):")
    #print(ctrl_field)
    print("MCS info (mod, code, shift_mod, shift_code) =", mcs_info)

    sync_and_control_data=symbol

    '''
    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(freq_vec))
    plt.title("Sync signal: Frequency domain")
    plt.grid(True)
    plt.show()

    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(symbol))
    plt.title("Sync Signal Time domain")
    plt.grid(True)
    plt.show()
    '''
	
    #print(ffd)

    '''
    decoded_results = decode_sync_and_control_ofdm_symbol(sync_and_control_data, FFT_SIZE, CP_LEN, k1, k2, u_sync=1, u_mod=1, u_code=1)
    print("Decoded Control Information:")
    print("Estimated Modulation:", decoded_results["estimated_modulation"])
    print("Estimated Coding Rate:", decoded_results["estimated_coding"])
    print("Estimated Mod Shift:", decoded_results["mod_shift"])
    print("Estimated Code Shift:", decoded_results["code_shift"])
    '''

    [K,used_subcarriers,unused_subcarriers]=compute_size_of_Tx_Rx_bits_and_subcarriers(max_used_subcarriers_excluding_pilot,modulation,code_rate,input_crc_size)

    print("KKKKKKKKKKKKKKKK=", K)        
    #num_subcarriers=used_subcarriers

    RU_SIZE=used_subcarriers # This is excluding pilots
    # APPLY CRC HERE
    print("RU Size=", RU_SIZE)

    # We create info bits, add CRC, encode => codeword => mod => map subcarriers => ...
    # We'll do 72 user bits => plus 16 CRC => 88 total
    user_bits= np.random.randint(0,2, K-input_crc_size, dtype=np.uint8)
    print("Bits without CRC=", len(user_bits))
    user_with_crc= add_crc16_bits(user_bits)
    # Encode
    final_info, codeword, mod_signal, H_qc= encode_qc3_11ax(user_with_crc, RU_SIZE, modulation, code_rate)
    print("Single user: final info len=", len(final_info), " codeword len=", len(codeword), " # symbols:", len(mod_signal))

    print("Size of CRC bits and coded bits. They must be the same (otherwise decoder will fail):", len(final_info), len(user_with_crc))
    if len(final_info)==len(user_with_crc) and (final_info==user_with_crc).all():
        print("*********** PERFECT SIZE: This setting is supported! ************")
    else:
        print("^^^^^^^^^^^^^^^^^^^^^This setting is not supported! Please choose another seting! STOP HERE!")

    # For example, allocate the RU block in the center of the FFT:
    #start_idx = (FFT_SIZE - RU_SIZE - len(pilotPosWithInRSCMinSCMax1)) // 2
    #RU_range = np.arange(start_idx, start_idx + RU_SIZE + len(pilotPosWithInRSCMinSCMax1))

    # Pilot positions are specified relative to the full FFT.
    # For instance, if you want pilots at absolute indices 60 and 190:
    #pilot_positions_NFFT = [60, 190]
    #pilot_value = 1 + 1j  # example pilot symbol

    data_subcarrier_positions=getDataSubcarrierIndexes1(FFT_SIZE,len(mod_signal),pilot_positions)
            
    freq_full = np.zeros(FFT_SIZE, dtype=np.complex64)

    freq_full[pilot_positions]=pilot_value
    freq_full[data_subcarrier_positions]=mod_signal

    print("Check the location of data sub-carriers")
    #print(ct1) 
    print("Check the size of the data subcarrier position and mod signal", len(mod_signal), len(data_subcarrier_positions))
    print("First 10 data sub-carroer positions=", data_subcarrier_positions[:10])
    
    plot_constellation(modulation, freq_full)
    #print(ffr)

    #print(ddf)
    print("Total number of transmitted bits=", K)
    print("Total number of used sub-carriers (Excluding pilots)=", used_subcarriers)
    print("Total number of unused sub-carriers (Excluding pilots)=", unused_subcarriers)

    print("Pilot positions (min and max):", min(pilot_positions), max(pilot_positions))
    #print("RU Range (min and max)", min(RU_range), max(RU_range))

    # Now freq_full is an array of length NFFT with pilots in the absolute positions,
    # and data symbols placed symmetrically in the RU block.

    # IFFT => CP => we get 1 symbol. Then we replicate KK times => pattern
    time_domain= np.fft.ifft(freq_full).astype(np.complex64)
    #low_papr_time_signal = iterative_clipping_and_filtering(freq_full,clipping_threshold,num_iterations,mask)
    #time_domain=low_papr_time_signal

    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.fft.fftshift(np.abs(freq_full)))
    plt.title("Tx Data Freq domain (DC at index 0)")
    plt.grid(True)
    plt.show()

    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(time_domain))
    plt.title("Tx Data Time domain")
    plt.grid(True)
    plt.show()

    #time_domain= np.fft.ifft(np.fft.fftshift(freq_full)).astype(np.complex64)
    cp= time_domain[-CP_LEN:]
    ofdm_symbol= np.concatenate((cp,time_domain))

    rx_symbol=ofdm_symbol

    print("Average Tx power per bit/symbol=", np.vdot(mod_signal,mod_signal)/len(mod_signal))
    print("Average Tx power per bit/symbol with IFFT (Power/bit reduces) =", np.vdot(time_domain,time_domain)/len(time_domain))
    
    k_sync = FFT_SIZE - (k1 + k2)
    # Now, process the received symbol.

    # It seems that DAC uses nice interpolation that can be tracked (or corrected by using timing estimation at Rx)
    # Timing estimation must be done at the receiver

    x_percent=99.99 # We need to clip it (to ensure unity max amplitude)
    sync_and_control_data = normalize_and_clip(sync_and_control_data, x_percent)
    ofdm_symbol = normalize_and_clip(ofdm_symbol, x_percent)

    # We replicate that symbol => single block (conatenate preamble + data)
    single_block= np.concatenate((sync_and_control_data,ofdm_symbol))
    #single_block=np.sqrt(FFT_SIZE)*single_block

    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(single_block))
    plt.title("Concatenated Block: Sync and Tx Data Time domain")
    plt.grid(True)
    plt.show()

    # We tile => single pattern => length = len(single_block)*KK
    
    temph1=np.random.randn(1)+1j*np.random.randn(1)
    temph1=1
    pattern= temph1*np.tile(single_block, KK)
    # If SIMULATE => AWGN => decode => else => TX/RX
    total_tx_samples= int(TT* tx_sampling_rate)
    repeats= total_tx_samples// len(pattern)
    print("Total number of repeated transmissions =", repeats)
    tx_waveform= np.tile(pattern, repeats).astype(np.complex64)
    chunk_size= YY*len(pattern)
    tx_waveform_sim= np.tile(pattern, YY).astype(np.complex64)

    #print(ddds)
    
    #out_folder= "temp80211ax"
    out_folder = r"C:\Users\tebogale\GD\pyFiles\temp80211ax"
    #out_folder = r"/home/uvify/WiFi80211ax/temp80211axData"

    #process_rx_chunk_spectrum(out_folder,FFT_SIZE,SIMULATE)
    #print(ddr)
    delete_folder_if_exists(out_folder)
    
    time.sleep(5)
    
    #print(ggf)

    print("Wait here!")
    # It is already shited where center = 0
    plt.figure()
    plt.plot(np.abs(single_block))
    plt.title("Concatenated Block: Sync and Tx Data Time domain")
    plt.grid(True)
    plt.show()


    if SIMULATE:
        # AWGN
        NR1=1
        bler1=[]
        for i in range(NR1):
            delete_folder_if_exists(out_folder)
            snr_lin= 10**(SNR_dB/10)
            #snr_lin= 10**(SNR_dB/10)
            fakeNoiseScale1=1 # Modified it intentionally
            noise_power= fakeNoiseScale1*1/(2*snr_lin) 
            noise_variance=noise_power
            # if 16QAM => complex => we do sqrt(noise_power/2) for real/imag
            # we add it to tx_waveform => then decode offline
            # We'll store the "received" as if it was a chunk
            sim_rx= tx_waveform_sim.copy()

            b_dict={'BPSK':1,'QPSK':2,'16QAM':4,'64QAM':6,'256QAM':8}
            if modulation not in b_dict:
                raise ValueError("Mod not supported!")
            b=b_dict[modulation]

            if b==1:
                noise = np.sqrt(noise_variance) * np.random.randn(len(sim_rx))
            else:
                noise = (np.sqrt(noise_variance/2)*np.random.randn(len(sim_rx)) +
                            1j*np.sqrt(noise_variance/2)*np.random.randn(len(sim_rx)))

            sim_rx+= noise

            # Example usage:
            #folder_to_delete = "path/to/your/folder"
            #delete_folder_if_exists(out_folder)

            #time.sleep(2)

            # chunk0 file
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            chunk_file= os.path.join(out_folder, "rx_chunk_0000.bin")
            sim_rx.tofile(chunk_file)
            # offline decode
            #G=2
            #print(fff)
            [tempPayload1,success1]=process_rx_chunk_practical(out_folder,FFT_SIZE,SIMULATE)
            bler1.append(success1)
        print("Successful Transmission rate = ", sum(bler1)/len(bler1))
    
    else:
        # actual hardware
        device= MultiUSRP("")
        device.set_tx_rate(tx_sampling_rate)
        device.set_rx_rate(rx_sampling_rate)
        device.set_tx_freq(TX_FREQ,0)
        device.set_rx_freq(RX_FREQ,0)
        device.set_tx_gain(TX_GAIN,0)
        device.set_rx_gain(RX_GAIN,0)
        tx_channel=[0]
        rx_channel=[0]
        
		#(device, chunk_size, RX_FREQ, rx_sampling_rate, rx_channel, RX_GAIN, TT, out_folder):

        #(device, tx_waveform, TX_FREQ, tx_sampling_rate, tx_channel, tx_gain, duration)
        
        #(device, tx_waveform, TT, TX_GAIN, [0])
        
        rx_thread= threading.Thread(target=rx_continuous, args=(device, chunk_size, RX_FREQ, rx_sampling_rate, rx_channel, RX_GAIN, TT, out_folder))
        tx_thread= threading.Thread(target=tx_continuous, args=(device, tx_waveform, TX_FREQ, tx_sampling_rate, tx_channel, TX_GAIN, TT))
        print("Starting continuous RX...")
        rx_thread.start()
        time.sleep(0.05)
        print("Starting continuous TX...")
        tx_thread.start()
        tx_thread.join()
        rx_thread.join()
        print("Continuous streaming complete. Chunks in folder:", out_folder)

        process_rx_chunk_practical(out_folder,FFT_SIZE,SIMULATE)
        #process_rx_chunk_simple(out_folder,FFT_SIZE,SIMULATE)
        
    print("Main done.")

if __name__=="__main__":
    main()
