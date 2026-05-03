"""Random / sequential sampler over the witek/ raw NCCA corpus.

The corpus stores 147,309,433 number-conserving Moore-9 rules across 12
worker files in 64-byte packed-LUT form (the same layout consumed by
ca_search/raw_reversibility_screen.py). This module exposes:

- WitekIndex: lazy directory of (file_path, n_records) pairs and a
  cumulative offset table, enabling O(log F) lookup of the (file,
  byte_offset) for any global record index k in [0, N_total).
- sample_random_lut_batch: vectorized random-without-replacement sampler
  that returns the LUTs as a (K, 512) uint8 numpy array, ready for the
  GPU pipeline. Uses multi-process I/O for K >= 1000 to overlap pread
  with decode.
- iter_sequential_lut_batches: stream the corpus in fixed-size chunks
  for full-catalogue scans.

Performance notes:
- For K = 10_000, the random sampler reads ~640 kB total off disk; even
  on cold cache it completes in <100 ms. Decoding (np.unpackbits + 0/1
  endpoints) is also <100 ms. So multiprocess is mostly useful for
  larger K (>= 100_000) or when the GPU pipeline can consume rules
  faster than a single thread can decode.
"""
from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_RECORD_BYTES = 64
RAW_RULE_BITS = 510  # x_1..x_510; x_0 = 0 and x_511 = 1 are fixed boundary
DEFAULT_WITEK_DIR = Path(__file__).resolve().parent.parent / "witek"


@dataclass(frozen=True)
class _FileSlice:
    path: Path
    n_records: int
    cum_start: int  # global index of the first record in this file


@dataclass
class WitekIndex:
    """Cumulative-offset index over a directory of worker_*.bin files."""

    files: list[_FileSlice]
    n_total: int

    @classmethod
    def open(cls, witek_dir: Path | str = DEFAULT_WITEK_DIR) -> "WitekIndex":
        witek_dir = Path(witek_dir)
        slices: list[_FileSlice] = []
        cum = 0
        for path in sorted(witek_dir.glob("worker_*.bin"),
                            key=lambda p: int(p.stem.split("_", 1)[1])):
            size = path.stat().st_size
            if size % RAW_RECORD_BYTES != 0:
                raise ValueError(f"{path} size {size} not multiple of {RAW_RECORD_BYTES}")
            n = size // RAW_RECORD_BYTES
            slices.append(_FileSlice(path=path, n_records=n, cum_start=cum))
            cum += n
        return cls(files=slices, n_total=cum)

    def locate(self, global_idx: int) -> tuple[Path, int]:
        """Return (file_path, byte_offset) for record `global_idx`."""
        if global_idx < 0 or global_idx >= self.n_total:
            raise IndexError(global_idx)
        # Binary search over cum_start.
        lo, hi = 0, len(self.files) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.files[mid].cum_start <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        slc = self.files[lo]
        within = global_idx - slc.cum_start
        return slc.path, within * RAW_RECORD_BYTES


def _decode_records_to_luts(records: np.ndarray) -> np.ndarray:
    """records: (K, 64) uint8.  Returns (K, 512) uint8 LUT bits."""
    unpacked = np.unpackbits(records, axis=1, bitorder="little")
    # Each row has 64 * 8 = 512 bits, but only the first 510 are valid;
    # we'd normally pad with x_0=0, x_511=1.  However the witek format
    # actually stores x_1..x_510 in the LOW 510 bits of the 64-byte
    # record (matching the unpackbits little-bit-order layout).  The
    # remaining 2 bits per record are zero by construction.
    luts = np.empty((len(records), 512), dtype=np.uint8)
    luts[:, 0] = 0
    luts[:, 1:511] = unpacked[:, :RAW_RULE_BITS]
    luts[:, 511] = 1
    return luts


def _read_offsets_from_file(path: Path, byte_offsets: np.ndarray) -> np.ndarray:
    """Read len(byte_offsets) records of RAW_RECORD_BYTES from path at the
    given byte offsets.  Returns (n, 64) uint8."""
    n = len(byte_offsets)
    out = np.empty((n, RAW_RECORD_BYTES), dtype=np.uint8)
    fd = os.open(path, os.O_RDONLY)
    try:
        for i, off in enumerate(byte_offsets):
            buf = os.pread(fd, RAW_RECORD_BYTES, int(off))
            out[i] = np.frombuffer(buf, dtype=np.uint8)
    finally:
        os.close(fd)
    return out


def sample_random_lut_batch(
    K: int,
    *,
    seed: int = 0,
    witek_dir: Path | str = DEFAULT_WITEK_DIR,
    index: WitekIndex | None = None,
    n_threads: int | None = None,
    return_global_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample K rules uniformly without replacement and return their LUTs.

    For K large (>= a few thousand) we group reads by file and use a thread
    pool to overlap pread calls across worker files.

    Returns (K, 512) uint8 numpy array.  If return_global_indices is True,
    also returns the (K,) int64 array of sampled global indices.
    """
    if index is None:
        index = WitekIndex.open(witek_dir)
    if K > index.n_total:
        raise ValueError(f"K={K} exceeds catalogue size {index.n_total}")
    rng = np.random.default_rng(seed)
    global_idx = rng.choice(index.n_total, size=K, replace=False)

    # Group by file (binary search + bucket sort).
    file_records: dict[int, list[tuple[int, int]]] = {}
    for j, gi in enumerate(global_idx):
        # Find file index
        lo, hi = 0, len(index.files) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if index.files[mid].cum_start <= gi:
                lo = mid
            else:
                hi = mid - 1
        within = int(gi - index.files[lo].cum_start)
        file_records.setdefault(lo, []).append((j, within))

    records = np.empty((K, RAW_RECORD_BYTES), dtype=np.uint8)

    def _read_one_file(file_idx: int):
        slc = index.files[file_idx]
        items = file_records[file_idx]
        # Sort by within-file offset for sequential pread
        items_sorted = sorted(items, key=lambda x: x[1])
        offsets = np.array([w * RAW_RECORD_BYTES for _, w in items_sorted], dtype=np.int64)
        rec_block = _read_offsets_from_file(slc.path, offsets)
        for (j, _), rec in zip(items_sorted, rec_block):
            records[j] = rec

    # Threads >= number of distinct files we touch (typically all 12)
    if n_threads is None:
        n_threads = min(12, len(file_records))
    if n_threads > 1 and len(file_records) > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(_read_one_file, file_records.keys()))
    else:
        for fi in file_records:
            _read_one_file(fi)

    luts = _decode_records_to_luts(records)
    if return_global_indices:
        return luts, global_idx
    return luts


def iter_sequential_lut_batches(
    batch_size: int,
    *,
    witek_dir: Path | str = DEFAULT_WITEK_DIR,
    index: WitekIndex | None = None,
    start_global_index: int = 0,
    end_global_index: int | None = None,
):
    """Yield (luts, global_indices) tuples in sequential corpus order.

    luts has shape (b, 512) uint8 where b <= batch_size.
    """
    if index is None:
        index = WitekIndex.open(witek_dir)
    if end_global_index is None:
        end_global_index = index.n_total

    cur = start_global_index
    while cur < end_global_index:
        n = min(batch_size, end_global_index - cur)
        # Read sequentially across file boundaries.
        records = np.empty((n, RAW_RECORD_BYTES), dtype=np.uint8)
        idxs = np.arange(cur, cur + n, dtype=np.int64)
        # Find file boundaries
        file_indices = np.zeros(n, dtype=np.int64)
        for i_rec, gi in enumerate(idxs):
            lo, hi = 0, len(index.files) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if index.files[mid].cum_start <= gi:
                    lo = mid
                else:
                    hi = mid - 1
            file_indices[i_rec] = lo
        # Group consecutive runs in the same file
        cur_pos = 0
        while cur_pos < n:
            f = file_indices[cur_pos]
            run_end = cur_pos
            while run_end < n and file_indices[run_end] == f:
                run_end += 1
            slc = index.files[f]
            first_within = idxs[cur_pos] - slc.cum_start
            count = run_end - cur_pos
            with slc.path.open("rb") as h:
                h.seek(int(first_within * RAW_RECORD_BYTES))
                blob = h.read(count * RAW_RECORD_BYTES)
            records[cur_pos:run_end] = np.frombuffer(blob, dtype=np.uint8).reshape(count, RAW_RECORD_BYTES)
            cur_pos = run_end
        luts = _decode_records_to_luts(records)
        yield luts, idxs
        cur += n


if __name__ == "__main__":
    # Smoke test: sample 1000 random rules, verify shapes and number-
    # conservation property (sum of LUT bits should be 256 = 512/2).
    import time
    idx = WitekIndex.open()
    print(f"corpus size: {idx.n_total} rules across {len(idx.files)} files")
    t0 = time.time()
    luts = sample_random_lut_batch(10_000, seed=1, index=idx)
    print(f"sampled 10k rules in {(time.time()-t0)*1000:.1f}ms")
    print(f"shape={luts.shape} dtype={luts.dtype}")
    sums = luts.sum(axis=1)
    print(f"NCCA sums: min={sums.min()} max={sums.max()} mean={sums.mean():.1f}")
    print(f"  (sum should be 256 for every NCCA: NC implies LUT half-1, half-0)")
    assert (sums == 256).all(), "non-NCCA rules in catalogue!"
    print("OK: all 10k sampled rules are NCCAs")
