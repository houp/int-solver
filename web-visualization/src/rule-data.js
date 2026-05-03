import { popcountMask } from "./lut.js";
import { enrichPropertyDefinition } from "./property-catalog.js";

export async function loadRuleDataset(url = "./data/rules-index.json") {
  const indexUrl = new URL(url, window.location.href);
  const response = await fetchJson(indexUrl);
  validateIndexDataset(response);

  const propertyCountMap = new Map(response.propertyCounts.map((item) => [item.bit, item.count]));
  const properties = response.properties
    .map((property) => enrichPropertyDefinition(property, propertyCountMap.get(property.bit) ?? 0))
    .sort((a, b) => a.label.localeCompare(b.label));

  const propertiesByBit = new Map(properties.map((property) => [property.bit, property]));
  const rules = response.rules.map((row) => enrichRule(row, response.ruleColumns));
  const rulesByStableId = new Map(rules.map((rule) => [rule.stableId, rule]));
  const shardPattern = String(response.detailShards?.pattern ?? "rule-details/shard-{shard}.json");
  const shardCache = new Map();

  async function ensureRuleDetail(stableId) {
    const rule = rulesByStableId.get(stableId);
    if (!rule) {
      throw new Error(`Unknown rule stable id: ${stableId}`);
    }
    if (typeof rule.lutHex === "string") {
      return rule;
    }
    const shardId = Number(rule.detailShard);
    await loadShard(shardId);
    return rulesByStableId.get(stableId);
  }

  async function loadShard(shardId) {
    if (!Number.isInteger(shardId) || shardId < 0) {
      throw new Error(`Invalid detail shard id: ${shardId}`);
    }
    if (!shardCache.has(shardId)) {
      shardCache.set(
        shardId,
        (async () => {
          const shardLabel = String(shardId).padStart(4, "0");
          const shardUrl = new URL(
            shardPattern.replace("{shard}", shardLabel),
            indexUrl,
          );
          const payload = await fetchJson(shardUrl);
          validateDetailShard(payload, shardId);
          for (const detail of payload.rules) {
            const rule = rulesByStableId.get(detail.stableId);
            if (rule) {
              rule.lutHex = detail.lutHex;
            }
          }
        })(),
      );
    }
    await shardCache.get(shardId);
  }

  return {
    summary: response.summary,
    properties,
    propertiesByBit,
    exactMaskCounts: response.exactMaskCounts,
    rules,
    rulesByStableId,
    ensureRuleDetail,
  };
}

async function fetchJson(url) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      "cache-control": "no-cache",
    },
  });
  if (!response.ok) {
    throw new Error(`Failed to load dataset: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function validateIndexDataset(dataset) {
  if (dataset.format !== "ca-rule-visualization-v2") {
    throw new Error(`Unsupported dataset format: ${dataset.format}`);
  }
  if (!Array.isArray(dataset.properties) || !Array.isArray(dataset.rules)) {
    throw new Error("Malformed dataset structure");
  }
  if (!Array.isArray(dataset.ruleColumns) || dataset.ruleColumns.length === 0) {
    throw new Error("Missing rule column schema");
  }
  if (!dataset.detailShards || typeof dataset.detailShards.pattern !== "string") {
    throw new Error("Missing detail shard metadata");
  }
}

function validateDetailShard(payload, shardId) {
  if (payload.format !== "ca-rule-detail-shard-v1") {
    throw new Error(`Unsupported detail shard format: ${payload.format}`);
  }
  if (payload.shard !== shardId || !Array.isArray(payload.rules)) {
    throw new Error(`Malformed detail shard payload for shard ${shardId}`);
  }
}

function enrichRule(row, ruleColumns) {
  if (!Array.isArray(row) || row.length !== ruleColumns.length) {
    throw new Error("Malformed compact rule row");
  }
  const compact = Object.fromEntries(ruleColumns.map((name, index) => [name, row[index]]));
  return {
    id: Number(compact.legacyIndex),
    legacyIndex: Number(compact.legacyIndex),
    stableIndex: Number(compact.stableIndex),
    stableId: String(compact.stableId),
    stableIdShort: String(compact.stableId).slice(0, 12),
    mask: Number(compact.mask),
    maskHex: `0x${Number(compact.mask).toString(16)}`,
    ones: Number(compact.ones),
    detailShard: Number(compact.detailShard),
    propertyCount: popcountMask(Number(compact.mask)),
    lutHex: null,
  };
}
