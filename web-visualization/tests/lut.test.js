import assert from "node:assert/strict";
import test from "node:test";

import {
  formatGroupedHex,
  hasMaskBit,
  hexToLutBits,
  neighborhoodIndex,
  popcountMask,
} from "../src/lut.js";

test("hexToLutBits decodes little-endian nibble groups", () => {
  const hex = "1".padEnd(128, "0");
  const bits = hexToLutBits(hex);
  assert.equal(bits.length, 512);
  assert.equal(bits[0], 1);
  assert.equal(bits[1], 0);
  assert.equal(bits[2], 0);
  assert.equal(bits[3], 0);
  assert.equal(bits[4], 0);
});

test("neighborhoodIndex matches x..c bit ordering", () => {
  assert.equal(neighborhoodIndex(0, 0, 0, 0, 0, 0, 0, 0, 0), 0);
  assert.equal(neighborhoodIndex(1, 0, 0, 0, 0, 0, 0, 0, 0), 256);
  assert.equal(neighborhoodIndex(0, 1, 0, 0, 0, 0, 0, 0, 0), 128);
  assert.equal(neighborhoodIndex(0, 0, 0, 0, 0, 0, 0, 0, 1), 1);
});

test("formatGroupedHex inserts lines and spaces", () => {
  const grouped = formatGroupedHex("0123456789abcdef".repeat(8), 8, 2);
  assert.match(grouped, /01234567/);
  assert.match(grouped, /\n/);
});

test("popcountMask counts set bits", () => {
  assert.equal(popcountMask(0), 0);
  assert.equal(popcountMask(0b101101), 4);
  assert.equal(popcountMask(2 ** 34 + 2 ** 31 + 2 ** 5 + 1), 4);
});

test("hasMaskBit works above 32 bits", () => {
  const mask = 2 ** 34 + 2 ** 31 + 2 ** 5 + 1;
  assert.equal(hasMaskBit(mask, 34), true);
  assert.equal(hasMaskBit(mask, 31), true);
  assert.equal(hasMaskBit(mask, 5), true);
  assert.equal(hasMaskBit(mask, 0), true);
  assert.equal(hasMaskBit(mask, 33), false);
});
