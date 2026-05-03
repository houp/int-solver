import assert from "node:assert/strict";
import test from "node:test";

import { neighborhoodIndex } from "../src/lut.js";
import { CASimulator } from "../src/simulator.js";

function buildRuleBits(predicate) {
  const bits = new Uint8Array(512);
  for (let index = 0; index < 512; index += 1) {
    const x = (index >> 8) & 1;
    const y = (index >> 7) & 1;
    const z = (index >> 6) & 1;
    const t = (index >> 5) & 1;
    const u = (index >> 4) & 1;
    const w = (index >> 3) & 1;
    const a = (index >> 2) & 1;
    const b = (index >> 1) & 1;
    const c = index & 1;
    bits[index] = predicate({ x, y, z, t, u, w, a, b, c });
  }
  return bits;
}

test("center-copy rule leaves the grid unchanged", () => {
  const simulator = new CASimulator(4, 4);
  simulator.grid.set([
    0, 1, 0, 1,
    1, 1, 0, 0,
    0, 0, 1, 1,
    1, 0, 1, 0,
  ]);
  simulator.population = 8;
  const before = Array.from(simulator.grid);
  const rule = buildRuleBits(({ u }) => u);
  simulator.step(rule);
  assert.deepEqual(Array.from(simulator.grid), before);
  assert.equal(simulator.generation, 1);
  assert.equal(simulator.population, 8);
});

test("left-neighbor copy respects periodic wraparound", () => {
  const simulator = new CASimulator(3, 1);
  simulator.grid.set([1, 0, 0]);
  simulator.population = 1;
  const rule = buildRuleBits(({ t }) => t);
  simulator.step(rule);
  assert.deepEqual(Array.from(simulator.grid), [0, 1, 0]);
});

test("neighborhoodIndex agrees with simulator bit ordering assumptions", () => {
  assert.equal(neighborhoodIndex(1, 0, 1, 0, 1, 0, 1, 0, 1), 341);
});
