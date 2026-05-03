import { neighborhoodIndex } from "./lut.js";

function clampInteger(value, min, max) {
  return Math.max(min, Math.min(max, Math.round(value)));
}

export class CASimulator {
  constructor(width = 64, height = 40) {
    this.resize(width, height);
  }

  resize(width, height) {
    const nextWidth = clampInteger(width, 1, 256);
    const nextHeight = clampInteger(height, 1, 256);
    const nextGrid = new Uint8Array(nextWidth * nextHeight);
    if (this.grid) {
      const copyWidth = Math.min(this.width, nextWidth);
      const copyHeight = Math.min(this.height, nextHeight);
      for (let y = 0; y < copyHeight; y += 1) {
        const sourceOffset = y * this.width;
        const targetOffset = y * nextWidth;
        nextGrid.set(this.grid.subarray(sourceOffset, sourceOffset + copyWidth), targetOffset);
      }
    }

    this.width = nextWidth;
    this.height = nextHeight;
    this.grid = nextGrid;
    this.buffer = new Uint8Array(nextWidth * nextHeight);
    this.generation = 0;
    this.population = sumBits(this.grid);
  }

  clear() {
    this.grid.fill(0);
    this.buffer.fill(0);
    this.generation = 0;
    this.population = 0;
  }

  fillRandom(probability = 0.35, random = Math.random) {
    const p = Math.max(0, Math.min(1, probability));
    for (let index = 0; index < this.grid.length; index += 1) {
      this.grid[index] = random() < p ? 1 : 0;
    }
    this.buffer.fill(0);
    this.generation = 0;
    this.population = sumBits(this.grid);
  }

  getCell(x, y) {
    return this.grid[y * this.width + x];
  }

  setCell(x, y, value) {
    const index = y * this.width + x;
    const previous = this.grid[index];
    const nextValue = value ? 1 : 0;
    if (previous === nextValue) {
      return previous;
    }
    this.grid[index] = nextValue;
    this.population += nextValue ? 1 : -1;
    return nextValue;
  }

  toggleCell(x, y) {
    const index = y * this.width + x;
    const nextValue = this.grid[index] ? 0 : 1;
    this.grid[index] = nextValue;
    this.population += nextValue ? 1 : -1;
    return nextValue;
  }

  step(lutBits, steps = 1) {
    const iterations = Math.max(1, Math.floor(steps));
    for (let i = 0; i < iterations; i += 1) {
      this.stepOnce(lutBits);
    }
  }

  stepOnce(lutBits) {
    const width = this.width;
    const height = this.height;
    const current = this.grid;
    const next = this.buffer;
    let population = 0;

    for (let y = 0; y < height; y += 1) {
      const north = y === 0 ? height - 1 : y - 1;
      const south = y === height - 1 ? 0 : y + 1;
      const northOffset = north * width;
      const rowOffset = y * width;
      const southOffset = south * width;

      for (let x = 0; x < width; x += 1) {
        const west = x === 0 ? width - 1 : x - 1;
        const east = x === width - 1 ? 0 : x + 1;
        const index = neighborhoodIndex(
          current[northOffset + west],
          current[northOffset + x],
          current[northOffset + east],
          current[rowOffset + west],
          current[rowOffset + x],
          current[rowOffset + east],
          current[southOffset + west],
          current[southOffset + x],
          current[southOffset + east],
        );
        const nextValue = lutBits[index];
        next[rowOffset + x] = nextValue;
        population += nextValue;
      }
    }

    this.grid = next;
    this.buffer = current;
    this.population = population;
    this.generation += 1;
  }
}

function sumBits(array) {
  let total = 0;
  for (const value of array) {
    total += value;
  }
  return total;
}
