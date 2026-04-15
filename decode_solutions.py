#!/usr/bin/env python3
import sys
from pathlib import Path

def iter_file(path, nvars):
    nwords = (nvars + 63) // 64
    block = nwords * 8
    with open(path, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                return
            words = [int.from_bytes(buf[i:i+8], "little") for i in range(0, block, 8)]
            bits = []
            for v in range(nvars):
                bits.append((words[v >> 6] >> (v & 63)) & 1)
            yield bits

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
