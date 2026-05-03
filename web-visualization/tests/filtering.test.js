import assert from "node:assert/strict";
import test from "node:test";

import {
  filterRules,
  findRuleBySid,
  normalizeSidQuery,
  paginateRules,
  ruleMatchesSelection,
} from "../src/filtering.js";

const rules = [
  {
    id: 0,
    stableIndex: 10,
    stableId: "deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678",
    mask: 0b0011,
    propertyCount: 2,
    ones: 10,
    searchText: "alpha 0x3",
  },
  {
    id: 1,
    stableIndex: 3,
    stableId: "cafebabe1234567890abcdef1234567890abcdef1234567890abcdef12345678",
    mask: 0b0010,
    propertyCount: 1,
    ones: 20,
    searchText: "beta 0x2",
  },
  {
    id: 2,
    stableIndex: 7,
    stableId: "cafe00001234567890abcdef1234567890abcdef1234567890abcdef12345678",
    mask: 0b0111,
    propertyCount: 3,
    ones: 15,
    searchText: "gamma 0x7",
  },
];

test("ruleMatchesSelection requires all selected bits", () => {
  assert.equal(ruleMatchesSelection(rules[0], [1]), true);
  assert.equal(ruleMatchesSelection(rules[0], [0]), true);
  assert.equal(ruleMatchesSelection(rules[0], [2]), false);
  assert.equal(ruleMatchesSelection(rules[2], [0, 1]), true);
});

test("filterRules combines mask selection and search", () => {
  const filtered = filterRules(rules, {
    selectedBits: [1],
    searchTerm: "beta",
    sortKey: "id-asc",
  });
  assert.deepEqual(
    filtered.map((rule) => rule.id),
    [1],
  );
});

test("paginateRules clamps page bounds", () => {
  const page = paginateRules(rules, 2, 10);
  assert.equal(page.totalPages, 2);
  assert.equal(page.pageIndex, 1);
  assert.deepEqual(
    page.items.map((rule) => rule.id),
    [2],
  );
});

test("id-asc sort follows stable index rather than legacy id", () => {
  const filtered = filterRules(rules, {
    selectedBits: [],
    searchTerm: "",
    sortKey: "id-asc",
  });
  assert.deepEqual(
    filtered.map((rule) => rule.stableIndex),
    [3, 7, 10],
  );
});

test("normalizeSidQuery strips prefix and non-hex characters", () => {
  assert.equal(normalizeSidQuery(" sid:DEAD-beef "), "deadbeef");
});

test("findRuleBySid resolves exact and prefix matches", () => {
  assert.equal(findRuleBySid(rules, "sid:deadbeef").status, "ok");
  assert.equal(findRuleBySid(rules, "cafebabe").rule.id, 1);
  assert.equal(findRuleBySid(rules, "cafe").status, "ambiguous");
  assert.equal(findRuleBySid(rules, "nope").status, "missing");
});

test("ruleMatchesSelection supports bits above 32", () => {
  const hiRule = {
    ...rules[0],
    mask: 2 ** 34 + 2 ** 31 + 2 ** 5 + 1,
  };
  assert.equal(ruleMatchesSelection(hiRule, [34, 31, 5, 0]), true);
  assert.equal(ruleMatchesSelection(hiRule, [33]), false);
});
