import { filterRules, findRuleBySid, paginateRules } from "./filtering.js";
import { GridView } from "./grid-view.js";
import { formatGroupedHex, hasMaskBit, hexToLutBits } from "./lut.js";
import { loadRuleDataset } from "./rule-data.js";
import { CASimulator } from "./simulator.js";

const state = {
  dataset: null,
  selectedPropertyBits: new Set(),
  selectedRuleStableId: null,
  selectedRuleBits: null,
  selectedRuleLoading: false,
  ruleLoadToken: 0,
  filteredRules: [],
  pageIndex: 0,
  pageSize: 50,
  sortKey: "propertyCount-desc",
  searchTerm: "",
  running: false,
  speed: 8,
};

const elements = {
  datasetSummary: document.querySelector("#dataset-summary"),
  selectionSummary: document.querySelector("#selection-summary"),
  openProperties: document.querySelector("#open-properties"),
  openPropertiesMeta: document.querySelector("#open-properties-meta"),
  openRules: document.querySelector("#open-rules"),
  openRulesMeta: document.querySelector("#open-rules-meta"),
  sidJumpForm: document.querySelector("#sid-jump-form"),
  sidJumpInput: document.querySelector("#sid-jump-input"),
  panelBackdrop: document.querySelector("#panel-backdrop"),
  propertiesPanel: document.querySelector("#properties-panel"),
  rulesPanel: document.querySelector("#rules-panel"),
  closeProperties: document.querySelector("#close-properties"),
  closeRules: document.querySelector("#close-rules"),
  propertyList: document.querySelector("#property-list"),
  clearProperties: document.querySelector("#clear-properties"),
  ruleSearch: document.querySelector("#rule-search"),
  sortKey: document.querySelector("#sort-key"),
  pageSize: document.querySelector("#page-size"),
  browserSummary: document.querySelector("#browser-summary"),
  ruleList: document.querySelector("#rule-list"),
  pageFirst: document.querySelector("#page-first"),
  pagePrev: document.querySelector("#page-prev"),
  pageNext: document.querySelector("#page-next"),
  pageLast: document.querySelector("#page-last"),
  pageStatus: document.querySelector("#page-status"),
  ruleDetail: document.querySelector("#rule-detail"),
  simGridWrap: document.querySelector(".sim-grid-wrap"),
  canvas: document.querySelector("#ca-canvas"),
  prevRule: document.querySelector("#prev-rule"),
  nextRule: document.querySelector("#next-rule"),
  gridWidth: document.querySelector("#grid-width"),
  gridHeight: document.querySelector("#grid-height"),
  resizeGrid: document.querySelector("#resize-grid"),
  randomProbability: document.querySelector("#random-probability"),
  randomProbabilityValue: document.querySelector("#random-probability-value"),
  fillRandom: document.querySelector("#fill-random"),
  clearGrid: document.querySelector("#clear-grid"),
  speed: document.querySelector("#speed"),
  speedValue: document.querySelector("#speed-value"),
  stepOnce: document.querySelector("#step-once"),
  toggleRun: document.querySelector("#toggle-run"),
  simStatus: document.querySelector("#sim-status"),
  jumpStatus: document.querySelector("#jump-status"),
  copyHex: document.querySelector("#copy-hex"),
};

const simulator = new CASimulator(
  Number(elements.gridWidth.value),
  Number(elements.gridHeight.value),
);
const gridView = new GridView(elements.canvas, elements.simGridWrap, handlePaintCell);

let animationFrame = null;
let lastFrameTime = 0;
let accumulatedTime = 0;

boot().catch((error) => {
  console.error(error);
  elements.browserSummary.innerHTML = `<div class="empty-state">Failed to initialize the app: ${escapeHtml(error.message)}</div>`;
});

async function boot() {
  wireControls();
  const dataset = await loadRuleDataset();
  state.dataset = dataset;
  renderDatasetSummary();
  renderPropertyFilters();
  applyFilters({ resetPage: true });
  simulator.fillRandom(Number(elements.randomProbability.value));
  renderSimulationStatus();
  redrawGrid();
}

function wireControls() {
  elements.openProperties.addEventListener("click", () => openPanel("properties"));
  elements.openRules.addEventListener("click", () => openPanel("rules"));
  elements.closeProperties.addEventListener("click", () => closePanels());
  elements.closeRules.addEventListener("click", () => closePanels());
  elements.panelBackdrop.addEventListener("click", () => closePanels());

  elements.clearProperties.addEventListener("click", () => {
    state.selectedPropertyBits.clear();
    renderPropertyFilters();
    applyFilters({ resetPage: true });
  });

  elements.sidJumpForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void jumpToSid(elements.sidJumpInput.value);
  });

  elements.ruleSearch.addEventListener("input", (event) => {
    state.searchTerm = event.target.value;
    applyFilters({ resetPage: true });
  });

  elements.sortKey.addEventListener("change", (event) => {
    state.sortKey = event.target.value;
    applyFilters({ resetPage: true });
  });

  elements.pageSize.addEventListener("change", (event) => {
    state.pageSize = Number(event.target.value);
    applyFilters({ resetPage: true });
  });

  elements.pageFirst.addEventListener("click", () => {
    state.pageIndex = 0;
    renderRuleBrowser();
  });
  elements.pagePrev.addEventListener("click", () => {
    state.pageIndex -= 1;
    renderRuleBrowser();
  });
  elements.pageNext.addEventListener("click", () => {
    state.pageIndex += 1;
    renderRuleBrowser();
  });
  elements.pageLast.addEventListener("click", () => {
    const totalPages = Math.max(1, Math.ceil(state.filteredRules.length / state.pageSize));
    state.pageIndex = totalPages - 1;
    renderRuleBrowser();
  });

  elements.prevRule.addEventListener("click", () => moveRuleSelection(-1));
  elements.nextRule.addEventListener("click", () => moveRuleSelection(1));

  elements.resizeGrid.addEventListener("click", () => {
    simulator.resize(Number(elements.gridWidth.value), Number(elements.gridHeight.value));
    redrawGrid();
    renderSimulationStatus();
  });

  elements.randomProbability.addEventListener("input", (event) => {
    elements.randomProbabilityValue.value = Number(event.target.value).toFixed(2);
  });

  elements.fillRandom.addEventListener("click", () => {
    simulator.fillRandom(Number(elements.randomProbability.value));
    redrawGrid();
    renderSimulationStatus();
  });

  elements.clearGrid.addEventListener("click", () => {
    simulator.clear();
    redrawGrid();
    renderSimulationStatus();
  });

  elements.speed.addEventListener("input", (event) => {
    state.speed = Number(event.target.value);
    elements.speedValue.value = `${state.speed} steps/s`;
  });

  elements.stepOnce.addEventListener("click", () => {
    pauseSimulation();
    stepSimulation(1);
  });

  elements.toggleRun.addEventListener("click", () => {
    if (state.running) {
      pauseSimulation();
    } else {
      startSimulation();
    }
  });

  elements.copyHex.addEventListener("click", async () => {
    const rule = getSelectedRule();
    if (!rule) {
      return;
    }
    try {
      if (!rule.lutHex) {
        await state.dataset.ensureRuleDetail(rule.stableId);
      }
      await navigator.clipboard.writeText(rule.lutHex);
      elements.copyHex.textContent = "Copied";
      window.setTimeout(() => {
        elements.copyHex.textContent = "Copy LUT Hex";
      }, 900);
    } catch (error) {
      console.warn("Clipboard write failed", error);
    }
  });

  window.addEventListener("resize", () => redrawGrid());
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePanels();
  });
}

function renderDatasetSummary() {
  const { summary, properties } = state.dataset;
  elements.datasetSummary.innerHTML = `
    <dl>
      <div>
        <dt>Base Rules</dt>
        <dd>${formatNumber(summary.totalBaseSolutions)}</dd>
      </div>
      <div>
        <dt>Listed Rules</dt>
        <dd>${formatNumber(summary.nonzeroMaskRuleCount)}</dd>
      </div>
      <div>
        <dt>Properties</dt>
        <dd>${formatNumber(properties.length)}</dd>
      </div>
      <div>
        <dt>LUT Size</dt>
        <dd>${summary.lutBits}</dd>
      </div>
    </dl>
  `;
}

async function jumpToSid(rawValue) {
  if (!state.dataset) {
    return;
  }
  const result = findRuleBySid(state.dataset.rules, rawValue);
  if (result.status === "empty") {
    setJumpStatus("Enter a stable rule id or prefix.", false);
    return;
  }
  if (result.status === "missing") {
    setJumpStatus(`No materialized rule matches sid:${result.prefix}.`, true);
    return;
  }
  if (result.status === "ambiguous") {
    setJumpStatus(`sid:${result.prefix} matches multiple rules. Paste a longer prefix.`, true);
    return;
  }

  state.selectedPropertyBits.clear();
  state.searchTerm = `sid:${result.rule.stableId}`;
  elements.ruleSearch.value = state.searchTerm;
  elements.sidJumpInput.value = `sid:${result.rule.stableIdShort}`;
  renderPropertyFilters();
  applyFilters({ resetPage: true });
  await selectRule(result.rule.stableId);
  setJumpStatus(`Selected sid:${result.rule.stableIdShort}.`, false);
}

function setJumpStatus(message, isError) {
  elements.jumpStatus.textContent = message;
  elements.jumpStatus.classList.toggle("is-error", Boolean(isError));
}

function renderPropertyFilters() {
  const selectedLabels = state.dataset.properties
    .filter((property) => state.selectedPropertyBits.has(property.bit))
    .map((property) => property.label);
  elements.selectionSummary.textContent =
    selectedLabels.length > 0
      ? `Active filters: ${selectedLabels.join(", ")}`
      : "No additional filters selected. Showing every materialized nonzero-mask rule.";
  elements.openPropertiesMeta.textContent = `${selectedLabels.length} active`;

  elements.propertyList.innerHTML = state.dataset.properties
    .map((property) => {
      const active = state.selectedPropertyBits.has(property.bit);
      return `
        <article class="property-card ${active ? "active" : ""}">
          <label>
            <input
              type="checkbox"
              data-property-bit="${property.bit}"
              ${active ? "checked" : ""}
            />
            <div>
              <div class="property-row">
                <div>
                  <h3>${escapeHtml(property.label)}</h3>
                </div>
                <span class="chip">${formatNumber(property.count)}</span>
              </div>
              <p>${escapeHtml(property.description)}</p>
              <div class="property-meta">
                <span class="chip secondary">${escapeHtml(property.category)}</span>
                <span class="chip mono">${escapeHtml(property.machineName)}</span>
              </div>
            </div>
          </label>
        </article>
      `;
    })
    .join("");

  for (const checkbox of elements.propertyList.querySelectorAll("input[type='checkbox']")) {
    checkbox.addEventListener("change", (event) => {
      const bit = Number(event.target.dataset.propertyBit);
      if (event.target.checked) {
        state.selectedPropertyBits.add(bit);
      } else {
        state.selectedPropertyBits.delete(bit);
      }
      renderPropertyFilters();
      applyFilters({ resetPage: true });
    });
  }
}

function applyFilters({ resetPage = false } = {}) {
  if (!state.dataset) {
    return;
  }
  if (resetPage) {
    state.pageIndex = 0;
  }
  const selectedMask = getSelectedMask();
  state.filteredRules = filterRules(state.dataset.rules, {
    selectedBits: [...state.selectedPropertyBits].sort((a, b) => a - b),
    searchTerm: state.searchTerm,
    sortKey: state.sortKey,
    properties: state.dataset.properties,
  });
  const selectedRule = getSelectedRule();
  if (!selectedRule || !state.filteredRules.some((rule) => rule.stableId === selectedRule.stableId)) {
    const first = state.filteredRules[0] ?? null;
    void selectRule(first?.stableId ?? null);
  } else {
    renderSelectedRule();
  }
  renderRuleBrowser();
}

function renderRuleBrowser() {
  const { items, totalPages, pageIndex } = paginateRules(
    state.filteredRules,
    state.pageSize,
    state.pageIndex,
  );
  state.pageIndex = pageIndex;

  const selectedMask = getSelectedMask();
  elements.browserSummary.innerHTML = `
    <span>Matching rules: <strong>${formatNumber(state.filteredRules.length)}</strong></span>
    <span class="chip mono">required mask ${selectedMask ? `0x${selectedMask.toString(16)}` : "none"}</span>
  `;
  elements.openRulesMeta.textContent = `${formatNumber(state.filteredRules.length)} matches`;

  if (items.length === 0) {
    elements.ruleList.innerHTML = `
      <div class="empty-state">
        No materialized rule satisfies the current property and search filters.
      </div>
    `;
  } else {
    elements.ruleList.innerHTML = items
      .map((rule) => renderRuleCard(rule, state.selectedRuleStableId === rule.stableId))
      .join("");
    for (const button of elements.ruleList.querySelectorAll("[data-rule-stable-id]")) {
      button.addEventListener("click", (event) => {
        void selectRule(event.currentTarget.dataset.ruleStableId);
        closePanels();
      });
    }
  }

  elements.pageStatus.textContent = `Page ${pageIndex + 1} of ${totalPages}`;
  elements.pageFirst.disabled = pageIndex === 0;
  elements.pagePrev.disabled = pageIndex === 0;
  elements.pageNext.disabled = pageIndex >= totalPages - 1;
  elements.pageLast.disabled = pageIndex >= totalPages - 1;
}

function renderRuleCard(rule, selected) {
  const detail = describeRule(rule);
  return `
    <article class="rule-card ${selected ? "selected" : ""}">
      <button type="button" class="rule-button" data-rule-stable-id="${rule.stableId}">
        <div class="rule-topline">
          <div>
            <p class="rule-title">${detail.displayName}</p>
            <p class="rule-subtitle">
              sid ${escapeHtml(rule.stableIdShort)} · mask ${escapeHtml(rule.maskHex)} · ${rule.propertyCount} properties · ${rule.ones} LUT ones
            </p>
          </div>
          <span class="chip">${rule.propertyCount}</span>
        </div>
        <div class="rule-property-list">
          ${detail.propertyLabels
            .map((label) => `<span class="chip">${escapeHtml(label)}</span>`)
            .join("")}
        </div>
      </button>
    </article>
  `;
}

async function selectRule(stableId) {
  state.selectedRuleStableId = stableId;
  const loadToken = ++state.ruleLoadToken;
  if (stableId == null) {
    state.selectedRuleBits = null;
    state.selectedRuleLoading = false;
    renderRuleBrowser();
    renderSelectedRule();
    return;
  }

  state.selectedRuleLoading = true;
  renderRuleBrowser();
  renderSelectedRule();
  const rule = getSelectedRule();
  if (!rule) {
    state.selectedRuleLoading = false;
    state.selectedRuleBits = null;
    renderSelectedRule();
    return;
  }
  if (!rule.lutHex) {
    try {
      await state.dataset.ensureRuleDetail(stableId);
    } catch (error) {
      if (loadToken !== state.ruleLoadToken) {
        return;
      }
      state.selectedRuleLoading = false;
      state.selectedRuleBits = null;
      elements.ruleDetail.innerHTML = `
        <div class="empty-state">
          Failed to load rule details: ${escapeHtml(error.message)}
        </div>
      `;
      elements.copyHex.disabled = true;
      return;
    }
  }
  if (loadToken !== state.ruleLoadToken || state.selectedRuleStableId !== stableId) {
    return;
  }
  state.selectedRuleLoading = false;
  state.selectedRuleBits = rule.lutHex ? hexToLutBits(rule.lutHex) : null;
  renderRuleBrowser();
  renderSelectedRule();
}

function renderSelectedRule() {
  const rule = getSelectedRule();
  if (!rule) {
    elements.ruleDetail.innerHTML = `
      <div class="empty-state">
        Select a rule from the browser to inspect it and run the automaton.
      </div>
    `;
    elements.copyHex.disabled = true;
    elements.prevRule.disabled = true;
    elements.nextRule.disabled = true;
    return;
  }
  const detail = describeRule(rule);
  elements.copyHex.disabled = !rule.lutHex || state.selectedRuleLoading;
  const currentIndex = getSelectedRuleIndex();
  const positionText =
    currentIndex >= 0 ? `${formatNumber(currentIndex + 1)} / ${formatNumber(state.filteredRules.length)}` : "n/a";
  elements.prevRule.disabled = currentIndex <= 0;
  elements.nextRule.disabled = currentIndex < 0 || currentIndex >= state.filteredRules.length - 1;
  elements.ruleDetail.innerHTML = `
    <div class="detail-grid">
      <dl class="detail-card">
        <dt>Stable Index</dt>
        <dd>${rule.stableIndex}</dd>
      </dl>
      <dl class="detail-card">
        <dt>Stable ID</dt>
        <dd class="mono">${escapeHtml(rule.stableIdShort)}</dd>
      </dl>
      <dl class="detail-card full">
        <dt>Stable ID (Full)</dt>
        <dd class="mono sid-value">${escapeHtml(rule.stableId)}</dd>
      </dl>
      <dl class="detail-card">
        <dt>Legacy Index</dt>
        <dd>${rule.legacyIndex}</dd>
      </dl>
      <dl class="detail-card">
        <dt>Mask</dt>
        <dd class="mono">${escapeHtml(rule.maskHex)}</dd>
      </dl>
      <dl class="detail-card">
        <dt>LUT Ones</dt>
        <dd>${formatNumber(rule.ones)}</dd>
      </dl>
      <dl class="detail-card">
        <dt>List Position</dt>
        <dd>${positionText}</dd>
      </dl>
      <dl class="detail-card full">
        <dt>Matched Properties</dt>
        <dd>${detail.propertyLabels.map(escapeHtml).join(", ")}</dd>
      </dl>
      <dl class="detail-card full">
        <dt>LUT Hex</dt>
        <dd><pre>${escapeHtml(detail.groupedHex ?? (state.selectedRuleLoading ? "Loading rule details…" : "Unavailable"))}</pre></dd>
      </dl>
    </div>
  `;
}

function startSimulation() {
  if (!state.selectedRuleBits) {
    return;
  }
  state.running = true;
  elements.toggleRun.textContent = "Pause";
  lastFrameTime = performance.now();
  accumulatedTime = 0;
  scheduleFrame();
}

function pauseSimulation() {
  state.running = false;
  elements.toggleRun.textContent = "Run";
  if (animationFrame != null) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }
}

function scheduleFrame() {
  animationFrame = requestAnimationFrame(handleAnimationFrame);
}

function handleAnimationFrame(timestamp) {
  if (!state.running || !state.selectedRuleBits) {
    animationFrame = null;
    return;
  }
  const frameDelta = timestamp - lastFrameTime;
  lastFrameTime = timestamp;
  accumulatedTime += frameDelta;
  const msPerStep = 1000 / state.speed;
  let steps = 0;
  while (accumulatedTime >= msPerStep && steps < 8) {
    stepSimulation(1, false);
    accumulatedTime -= msPerStep;
    steps += 1;
  }
  redrawGrid();
  renderSimulationStatus();
  scheduleFrame();
}

function stepSimulation(steps = 1, redraw = true) {
  if (!state.selectedRuleBits) {
    return;
  }
  simulator.step(state.selectedRuleBits, steps);
  if (redraw) {
    redrawGrid();
    renderSimulationStatus();
  }
}

function renderSimulationStatus() {
  const density = simulator.grid.length > 0 ? simulator.population / simulator.grid.length : 0;
  elements.simStatus.innerHTML = `
    Generation <strong>${formatNumber(simulator.generation)}</strong>
    · Population <strong>${formatNumber(simulator.population)}</strong>
    · Density <strong>${density.toFixed(3)}</strong>
    · Grid <strong>${simulator.width} × ${simulator.height}</strong>
  `;
}

function redrawGrid() {
  gridView.syncCanvasSize(simulator);
  gridView.render(simulator);
}

function openPanel(which) {
  const showProperties = which === "properties";
  const showRules = which === "rules";
  elements.propertiesPanel.classList.toggle("is-open", showProperties);
  elements.rulesPanel.classList.toggle("is-open", showRules);
  elements.propertiesPanel.setAttribute("aria-hidden", String(!showProperties));
  elements.rulesPanel.setAttribute("aria-hidden", String(!showRules));
  elements.panelBackdrop.hidden = false;
  document.body.classList.add("panel-open");
}

function closePanels() {
  elements.propertiesPanel.classList.remove("is-open");
  elements.rulesPanel.classList.remove("is-open");
  elements.propertiesPanel.setAttribute("aria-hidden", "true");
  elements.rulesPanel.setAttribute("aria-hidden", "true");
  elements.panelBackdrop.hidden = true;
  document.body.classList.remove("panel-open");
}

function handlePaintCell(x, y, paintValue) {
  const nextValue = paintValue == null ? simulator.toggleCell(x, y) : simulator.setCell(x, y, paintValue);
  redrawGrid();
  renderSimulationStatus();
  return nextValue;
}

function moveRuleSelection(delta) {
  if (state.filteredRules.length === 0) {
    return;
  }
  const currentIndex = getSelectedRuleIndex();
  const baseIndex = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = Math.min(state.filteredRules.length - 1, Math.max(0, baseIndex + delta));
  if (nextIndex === currentIndex) {
    return;
  }
  void selectRule(state.filteredRules[nextIndex].stableId);
}

function getSelectedMask() {
  let mask = 0;
  for (const bit of state.selectedPropertyBits) {
    mask += 2 ** bit;
  }
  return mask;
}

function getSelectedRule() {
  if (state.selectedRuleStableId == null) {
    return null;
  }
  return state.dataset.rulesByStableId.get(state.selectedRuleStableId) ?? null;
}

function getSelectedRuleIndex() {
  if (state.selectedRuleStableId == null) {
    return -1;
  }
  return state.filteredRules.findIndex((rule) => rule.stableId === state.selectedRuleStableId);
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function truncateGroupedHex(groupedHex, maxLines) {
  const lines = groupedHex.split("\n");
  if (lines.length <= maxLines) {
    return groupedHex;
  }
  return `${lines.slice(0, maxLines).join("\n")}\n…`;
}

function describeRule(rule) {
  const matchedProperties = [];
  for (const [bit, property] of state.dataset.propertiesByBit) {
    if (hasMaskBit(rule.mask, bit)) {
      matchedProperties.push(property);
    }
  }
  return {
    displayName: `Rule ${rule.stableIndex}`,
    propertyLabels: matchedProperties.map((property) => property.label),
    groupedHex: rule.lutHex ? formatGroupedHex(rule.lutHex) : null,
  };
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
