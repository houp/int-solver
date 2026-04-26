const NIBBLE_BITS = {
  "0": [0, 0, 0, 0],
  "1": [1, 0, 0, 0],
  "2": [0, 1, 0, 0],
  "3": [1, 1, 0, 0],
  "4": [0, 0, 1, 0],
  "5": [1, 0, 1, 0],
  "6": [0, 1, 1, 0],
  "7": [1, 1, 1, 0],
  "8": [0, 0, 0, 1],
  "9": [1, 0, 0, 1],
  a: [0, 1, 0, 1],
  b: [1, 1, 0, 1],
  c: [0, 0, 1, 1],
  d: [1, 0, 1, 1],
  e: [0, 1, 1, 1],
  f: [1, 1, 1, 1],
};

export function hexToLutBits(lutHex) {
  if (lutHex.length !== 128) {
    throw new Error(`Expected 128 hex digits, received ${lutHex.length}`);
  }
  const bits = new Uint8Array(512);
  let offset = 0;
  for (const ch of lutHex.toLowerCase()) {
    const nibble = NIBBLE_BITS[ch];
    if (!nibble) {
      throw new Error(`Invalid LUT hex digit: ${ch}`);
    }
    bits[offset] = nibble[0];
    bits[offset + 1] = nibble[1];
    bits[offset + 2] = nibble[2];
    bits[offset + 3] = nibble[3];
    offset += 4;
  }
  return bits;
}

export function neighborhoodIndex(x, y, z, t, u, w, a, b, c) {
  return (
    (x << 8) |
    (y << 7) |
    (z << 6) |
    (t << 5) |
    (u << 4) |
    (w << 3) |
    (a << 2) |
    (b << 1) |
    c
  );
}

export function formatGroupedHex(lutHex, groupSize = 8, groupsPerLine = 4) {
  const groups = [];
  for (let index = 0; index < lutHex.length; index += groupSize) {
    groups.push(lutHex.slice(index, index + groupSize));
  }
  const lines = [];
  for (let index = 0; index < groups.length; index += groupsPerLine) {
    lines.push(groups.slice(index, index + groupsPerLine).join(" "));
  }
  return lines.join("\n");
}

export function popcountMask(mask) {
  let remaining = Math.max(0, Math.trunc(mask));
  let count = 0;
  while (remaining > 0) {
    count += remaining % 2;
    remaining = Math.floor(remaining / 2);
  }
  return count;
}

export function hasMaskBit(mask, bit) {
  if (!Number.isInteger(bit) || bit < 0) {
    return false;
  }
  const divisor = 2 ** bit;
  return Math.floor(Math.trunc(mask) / divisor) % 2 === 1;
}
