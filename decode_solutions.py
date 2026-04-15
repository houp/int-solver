#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    zstd = None

def decode_bits(buf, nvars):
    nwords = (nvars + 63) // 64
    words = [int.from_bytes(buf[i:i+8], "little") for i in range(0, nwords * 8, 8)]
    bits = []
    for v in range(nvars):
        bits.append((words[v >> 6] >> (v & 63)) & 1)
    return bits

def iter_file(path, nvars):
    nwords = (nvars + 63) // 64
    sol_size = nwords * 8
    
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"ZSD1":
            if zstd is None:
                print(f"Error: {path} is compressed but 'zstandard' library is not installed.", file=sys.stderr)
                return
            
            dctx = zstd.ZstdDecompressor()
            while True:
                sz_buf = f.read(4)
                if not sz_buf:
                    break
                sz = int.from_bytes(sz_buf, "little")
                compressed = f.read(sz)
                decompressed = dctx.decompress(compressed)
                for i in range(0, len(decompressed), sol_size):
                    yield decode_bits(decompressed[i:i+sol_size], nvars)
        else:
            # Not compressed, or unknown magic. Assume uncompressed and seek back.
            f.seek(0)
            while True:
                buf = f.read(sol_size)
                if not buf:
                    break
                yield decode_bits(buf, nvars)

def main():
    if len(sys.argv) != 3:
        print("usage: decode_solutions.py <solutions_dir_or_file> <nvars>")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    nvars = int(sys.argv[2])
    files = [path] if path.is_file() else sorted(path.glob("*.bin"))
    for fp in files:
        for bits in iter_file(fp, nvars):
            print(" ".join(map(str, bits)))

if __name__ == "__main__":
    main()
