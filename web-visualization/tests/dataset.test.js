import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

test("generated dataset has expected top-level structure", async () => {
  const raw = await readFile(
    new URL("../data/rules-index.json", import.meta.url),
    "utf8",
  );
  const dataset = JSON.parse(raw);
  assert.equal(dataset.format, "ca-rule-visualization-v2");
  assert.equal(dataset.rules.length, dataset.summary.nonzeroMaskRuleCount);
  assert.ok(dataset.rules.length > 0);
  assert.equal(dataset.properties.length, dataset.propertyCounts.length);
  assert.equal(dataset.summary.lutBits, 512);
  assert.deepEqual(dataset.ruleColumns, [
    "legacyIndex",
    "stableIndex",
    "stableId",
    "mask",
    "ones",
    "detailShard",
  ]);
  assert.ok(dataset.rules.every((rule) => Array.isArray(rule) && rule.length === dataset.ruleColumns.length));
  assert.ok(dataset.rules.every((rule) => typeof rule[2] === "string" && rule[2].length === 64));
  assert.ok(dataset.rules.every((rule) => Number.isInteger(rule[1])));
  assert.ok(dataset.rules.every((rule) => Number.isInteger(rule[5])));
  assert.ok(typeof dataset.detailShards.pattern === "string");
});

test("detail shard contains LUT hex payloads", async () => {
  const raw = await readFile(
    new URL("../data/rules-index.json", import.meta.url),
    "utf8",
  );
  const dataset = JSON.parse(raw);
  const shard = String(dataset.rules[0][5]).padStart(4, "0");
  const detailRaw = await readFile(
    new URL(`../data/rule-details/shard-${shard}.json`, import.meta.url),
    "utf8",
  );
  const detail = JSON.parse(detailRaw);
  assert.equal(detail.format, "ca-rule-detail-shard-v1");
  assert.ok(detail.rules.length > 0);
  assert.ok(detail.rules.every((rule) => typeof rule.lutHex === "string" && rule.lutHex.length === 128));
});
