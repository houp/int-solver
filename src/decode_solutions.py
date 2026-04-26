#!/usr/bin/env python3
import sys
import struct
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

def decode_mask(buf, nprops):
    value = int.from_bytes(buf, "little")
    return value, [(value >> i) & 1 for i in range(nprops)]

def iter_file(path, nvars):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"CPM1":
            header_rest = f.read(28)
            if len(header_rest) != 28:
                raise RuntimeError(f"{path} has a truncated CPM1 header")
            version, header_nvars, nprops, mask_bytes, solution_bytes, total_records = struct.unpack(
                "<IIIIIQ", header_rest
            )
            if version != 1:
                raise RuntimeError(f"Unsupported CPM1 version {version} in {path}")
            if nvars is not None and header_nvars != nvars:
                raise RuntimeError(
                    f"{path} header says nvars={header_nvars}, but decoder was asked for nvars={nvars}"
                )
            nvars = header_nvars
            expected_solution_bytes = ((nvars + 63) // 64) * 8
            if solution_bytes != expected_solution_bytes:
                raise RuntimeError(
                    f"{path} has solution_bytes={solution_bytes}, expected {expected_solution_bytes}"
                )
            while True:
                mask_buf = f.read(mask_bytes)
                if not mask_buf:
                    break
                if len(mask_buf) != mask_bytes:
                    raise RuntimeError(f"{path} has a truncated mask record")
                sol_buf = f.read(solution_bytes)
                if len(sol_buf) != solution_bytes:
                    raise RuntimeError(f"{path} has a truncated solution record")
                mask_value, mask_bits = decode_mask(mask_buf, nprops)
                yield {
                    "mask_value": mask_value,
                    "mask_bits": mask_bits,
                    "bits": decode_bits(sol_buf, nvars),
                    "total_records": total_records,
                }
        elif magic == b"ZSD1":
            if nvars is None:
                raise RuntimeError(f"{path} needs an explicit nvars argument for legacy ZSD1 decoding")
            nwords = (nvars + 63) // 64
            sol_size = nwords * 8
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
                    yield {"mask_value": None, "mask_bits": None, "bits": decode_bits(decompressed[i:i+sol_size], nvars)}
        else:
            if nvars is None:
                raise RuntimeError(f"{path} needs an explicit nvars argument for legacy raw decoding")
            nwords = (nvars + 63) // 64
            sol_size = nwords * 8
            # Not compressed, or unknown magic. Assume uncompressed and seek back.
            f.seek(0)
            while True:
                buf = f.read(sol_size)
                if not buf:
                    break
                yield {"mask_value": None, "mask_bits": None, "bits": decode_bits(buf, nvars)}

def main():
    if len(sys.argv) not in (2, 3):
        print("usage: decode_solutions.py <solutions_dir_or_file> [nvars]")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    nvars = int(sys.argv[2]) if len(sys.argv) == 3 else None
    files = [path] if path.is_file() else sorted(path.glob("*.bin"))
    for fp in files:
        for item in iter_file(fp, nvars):
            if item["mask_bits"] is None:
                print(" ".join(map(str, item["bits"])))
            else:
                mask_str = "".join(map(str, item["mask_bits"]))
                print(f"mask={item['mask_value']} mask_bits={mask_str} " + " ".join(map(str, item["bits"])))

if __name__ == "__main__":
    main()
