# UAT-project



# OFDM Tx/Rx Coded Simulation (single-user)

Short README for the simulation/processing script:

UAT.py

**Description**
- Single-user OFDM/OFDM-A style transmitter + receiver simulation with QC-LDPC toy encoders, CRC-16, and symbol-level processing.
- Implements sync/control symbol construction (Zadoff–Chu), LS/MMSE channel estimation, modulation/demodulation (BPSK/QPSK/16QAM/64QAM/256QAM), iterative clipping/filtering (PAPR reduction), and decoding using an LSD-like LDPC approach.
- Can run purely in simulation (AWGN loopback) or drive real hardware via UHD/USRP.

**Key features**
- CRC16 utilities and helpers to append/verify 16-bit CRC
- QC-LDPC toy base matrices (multiple rates) and encode/decode helpers
- Sync/control OFDM symbol generation + detection (synchronization, CFO estimation)
- Per-subcarrier equalization and MMSE-style channel estimation from pilots
- Utilities for power/SNR estimation and simple PAPR reduction (iterative clipping and filtering)
- Offline chunk-based RX processing and example continuous TX/RX threads for UHD

**Requirements**
- Python 3.8+ (tested with CPython)
- Core Python packages: numpy, scipy, matplotlib, numba
- Optional/hardware: uhd (pyuhd) and a supported USRP device when using real TX/RX
- FEC helper: an `ldpc` package is imported (project includes references to `ldpc.bplsd_decoder`). If you don't have it, running simulation-only paths still uses some LDPC helpers — install or provide that package.

Install typical dependencies with pip:
# install UHD/pyuhd if you plan to use USRP hardware (platform-specific)
```

If `ldpc` is not available via pip for your environment, add the local `ldpc` package used by this repository to `PYTHONPATH`.

**Quick start / Usage**
- Edit the top-level options inside the script or run it directly:

```
python UAT.py
```

- Toggle simulation vs hardware by changing `SIMULATE = True` near `main()`:
  - `SIMULATE = True` — script runs offline AWGN loopback and writes/reads chunk files.
  - `SIMULATE = False` — script will attempt to use UHD/USRP (`MultiUSRP`) and real TX/RX.

**Important configuration points**
- FFT size and CP: change `FFT_SIZE` and `commonTxRxParameters(FFT_SIZE)` for sampling rate, CP length, and pilot positions.
- `out_folder` controls where RX chunks are saved/loaded (default in the script: a local `temp80211ax` path).
- `TX_FREQ`, `RX_FREQ`, `TX_GAIN`, `RX_GAIN` are configured in `commonTxRxParameters()` and used when hardware mode is active.
**Files produced / outputs**
- When simulating, the script writes `rx_chunk_*.bin` files into `out_folder` and processes them with offline decoders.
- The code prints status lines about estimated SNR, candidate sync indices, CRC checks, and success rate (bler).

**Notes & caveats**
- The script includes many experimental /

```
pip install numpy scipy matplotlib numba
