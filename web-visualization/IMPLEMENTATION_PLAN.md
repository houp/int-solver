# Web Visualization Implementation Plan

## Goal

Build a high-quality, fully client-side desktop web application for exploring the `7,089` nonzero-mask number-conserving rules produced by the solver. The app should let users:

- filter rules by any combination of CA properties,
- inspect rule LUTs in hexadecimal form,
- select a rule and simulate it on a periodic grid,
- edit or randomize the initial configuration,
- run, pause, and step the automaton.

All code, assets, generated datasets, tests, and documentation live under `web-visualization/`.

## Constraints And Design Choices

- No server logic. The app must run as static files in a browser.
- Use the already-generated solver outputs as the source of truth:
  - `full_property_panel_nonzero.bin`
  - `full_property_panel_nonzero.json`
- Precompute a client-friendly dataset so the browser does not need to parse large solver binaries on startup.
- Keep the code modular and testable without depending on a backend.
- Optimize for desktop browsers and readability first; keep the architecture extensible.

## Architecture

### Data Pipeline

1. Parse the masked solver output and metadata.
2. Reconstruct each rule’s full 512-entry LUT:
   - `x_0 = 0`
   - `x_1..x_510` from the stored solver record
   - `x_511 = 1`
3. Encode the LUT as a fixed 128-hex-digit string.
4. Emit a static JSON dataset containing:
   - property definitions and bit positions,
   - exact mask counts,
   - rule list with masks and LUT hex strings,
   - summary counts for the nonzero subset.

### Frontend

1. Static single-page app using browser ES modules.
2. Canvas-based grid rendering for efficient simulation and editing.
3. Modular state management:
   - dataset loading,
   - property filtering,
   - selected rule,
   - simulation state,
   - UI controls.
4. Clear layout:
   - property filter panel,
   - rule list panel,
   - simulation and controls panel,
   - rule details panel.

### Simulation

1. Periodic boundary conditions.
2. Neighborhood bit order identical to the generator:
   - `(x, y, z, t, u, w, a, b, c) -> index`
3. Double-buffered update step using typed arrays.
4. Manual painting and random initialization supported.

## Step Breakdown

### Step 1. Project Setup

- Create `web-visualization/` structure.
- Add `package.json` with lightweight scripts for tests and local static serving.
- Add this plan file.

### Step 2. Data Conversion

- Implement a Python script that reads the solver artifacts and writes:
  - `web-visualization/data/rules-dataset.json`
- Validate:
  - record count equals `written_solution_records`,
  - nonzero mask totals match the metadata,
  - every LUT string has length `128`.

### Step 3. Core App Modules

- `rule-data.js`: dataset loading and lookup helpers.
- `filtering.js`: property-mask filtering.
- `lut.js`: hex/LUT conversion helpers.
- `simulator.js`: CA stepping logic and periodic boundaries.
- `grid-view.js`: canvas rendering and interaction.
- `app.js`: UI orchestration.

### Step 4. User Interface

- Property checklist with counts and descriptions.
- Rule list with:
  - stable sort,
  - visible mask summary,
  - grouped hex LUT display,
  - selection state.
- Simulation controls:
  - width,
  - height,
  - random fill probability,
  - clear,
  - fill random,
  - step,
  - run/pause,
  - speed indicator,
  - generation and population counters.

### Step 5. Testing

- Node tests for:
  - dataset structure,
  - mask filtering behavior,
  - LUT decoding,
  - periodic simulation step correctness.
- Data generation validation script run after dataset build.

### Step 6. Documentation

- Add `web-visualization/README.md` with:
  - purpose,
  - file layout,
  - dataset format,
  - how to run locally,
  - simulation conventions,
  - future extension points.

## Quality Bar

- Filtering must remain responsive on all `7,089` rules.
- Simulation should run smoothly on moderate desktop grids.
- The code should clearly separate data, simulation, rendering, and UI logic.
- The dataset format should be easy to regenerate if solver artifacts change.
