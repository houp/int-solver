export class GridView {
  constructor(canvas, viewport, onPaint) {
    this.canvas = canvas;
    this.viewport = viewport;
    this.context = canvas.getContext("2d");
    this.context.imageSmoothingEnabled = false;
    this.offscreen = document.createElement("canvas");
    this.offscreenContext = this.offscreen.getContext("2d");
    this.offscreenContext.imageSmoothingEnabled = false;
    this.imageData = null;
    this.onPaint = onPaint;
    this.dragging = false;
    this.paintValue = 1;
    this.lastCell = null;
    this.cellSize = 1;
    this.displayWidth = 0;
    this.displayHeight = 0;
    this.devicePixelRatio = 1;

    canvas.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    canvas.addEventListener("pointerup", () => this.endDrag());
    canvas.addEventListener("pointerleave", () => this.endDrag());
    canvas.addEventListener("pointercancel", () => this.endDrag());
  }

  syncCanvasSize(simulator) {
    const { width, height } = simulator;
    const viewportWidth = Math.max(1, Math.floor(this.viewport.clientWidth));
    const viewportHeight = Math.max(1, Math.floor(this.viewport.clientHeight));
    const fitByWidth = Math.max(1, Math.floor(viewportWidth / width));
    const fitByHeight = Math.max(1, Math.floor(viewportHeight / height));
    const cellSize = Math.max(1, Math.min(16, fitByWidth, fitByHeight));
    const displayWidth = width * cellSize;
    const displayHeight = height * cellSize;
    const dpr = window.devicePixelRatio || 1;
    const backingWidth = displayWidth * dpr;
    const backingHeight = displayHeight * dpr;

    if (
      this.cellSize === cellSize &&
      this.displayWidth === displayWidth &&
      this.displayHeight === displayHeight &&
      this.devicePixelRatio === dpr &&
      this.canvas.width === backingWidth &&
      this.canvas.height === backingHeight
    ) {
      return false;
    }

    this.cellSize = cellSize;
    this.displayWidth = displayWidth;
    this.displayHeight = displayHeight;
    this.devicePixelRatio = dpr;
    this.canvas.width = backingWidth;
    this.canvas.height = backingHeight;
    this.canvas.style.width = `${displayWidth}px`;
    this.canvas.style.height = `${displayHeight}px`;
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.context.imageSmoothingEnabled = false;
    return true;
  }

  render(simulator) {
    const { width, height, grid } = simulator;
    if (this.offscreen.width !== width || this.offscreen.height !== height) {
      this.offscreen.width = width;
      this.offscreen.height = height;
      this.imageData = this.offscreenContext.createImageData(width, height);
    }

    const pixels = this.imageData.data;
    for (let index = 0; index < grid.length; index += 1) {
      const value = grid[index] ? 238 : 17;
      const alpha = 255;
      const pixelOffset = index * 4;
      pixels[pixelOffset] = value;
      pixels[pixelOffset + 1] = grid[index] ? 247 : 24;
      pixels[pixelOffset + 2] = grid[index] ? 243 : 31;
      pixels[pixelOffset + 3] = alpha;
    }
    this.offscreenContext.putImageData(this.imageData, 0, 0);

    const context = this.context;
    context.clearRect(0, 0, this.displayWidth, this.displayHeight);
    context.drawImage(this.offscreen, 0, 0, this.displayWidth, this.displayHeight);
    drawGridLines(context, width, height, this.cellSize, this.displayWidth, this.displayHeight);
  }

  handlePointerDown(event) {
    const cell = this.eventToCell(event);
    if (!cell) {
      return;
    }
    this.dragging = true;
    this.canvas.setPointerCapture(event.pointerId);
    this.lastCell = null;
    this.paintValue = this.onPaint(cell.x, cell.y, null);
    this.lastCell = cell;
  }

  handlePointerMove(event) {
    if (!this.dragging) {
      return;
    }
    const cell = this.eventToCell(event);
    if (!cell) {
      return;
    }
    if (this.lastCell && this.lastCell.x === cell.x && this.lastCell.y === cell.y) {
      return;
    }
    this.onPaint(cell.x, cell.y, this.paintValue);
    this.lastCell = cell;
  }

  endDrag() {
    this.dragging = false;
    this.lastCell = null;
  }

  eventToCell(event) {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return null;
    }
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || x >= rect.width || y < 0 || y >= rect.height) {
      return null;
    }
    const cellX = Math.min(this.offscreen.width - 1, Math.floor(x / this.cellSize));
    const cellY = Math.min(this.offscreen.height - 1, Math.floor(y / this.cellSize));
    return { x: cellX, y: cellY };
  }
}

function drawGridLines(context, width, height, cellSize, canvasWidth, canvasHeight) {
  if (cellSize < 6) {
    return;
  }
  context.save();
  context.strokeStyle = "rgba(255,255,255,0.08)";
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 1; x < width; x += 1) {
    const pos = x * cellSize + 0.5;
    context.moveTo(pos, 0);
    context.lineTo(pos, canvasHeight);
  }
  for (let y = 1; y < height; y += 1) {
    const pos = y * cellSize + 0.5;
    context.moveTo(0, pos);
    context.lineTo(canvasWidth, pos);
  }
  context.stroke();
  context.restore();
}
