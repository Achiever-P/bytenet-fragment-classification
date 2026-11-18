"""Byte2Image: bit-shift stack (Eq. 2) and intrabyte n-gram (Eq. 3)."""
import numpy as np

def bit_shift_stack(byte_seq):
    byte_seq = np.asarray(byte_seq, dtype=np.uint8)
    x_in = np.empty((byte_seq.shape[0], 8), dtype=np.uint8)
    cur = byte_seq.copy()
    for i in range(8):
        x_in[:, i] = cur
        cur = ((cur.astype(np.uint16) << 1) & 0xFF).astype(np.uint8)
    return x_in

def intrabyte_ngram(x_in, n=4):
    ns = x_in.shape[0]
    h = ns - n + 1
    w = np.lib.stride_tricks.sliding_window_view(x_in, (n, 8))
    return w[:, 0, :, :].reshape(h, n * 8)

def byte2image(byte_seq, n=4):
    return intrabyte_ngram(bit_shift_stack(byte_seq), n=n).astype(np.float32)

def byte_stats(byte_seq):
    """Position-invariant statistics: 256-bin histogram (sums to 1) and Shannon entropy / 8."""
    b = np.asarray(byte_seq, dtype=np.uint8)
    hist = np.bincount(b, minlength=256).astype(np.float32) / len(b)
    nz = hist[hist > 0]
    ent = float(-(nz * np.log2(nz)).sum()) / 8.0
    return np.concatenate([hist * 16.0, [ent]]).astype(np.float32)
