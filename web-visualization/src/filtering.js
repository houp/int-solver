function compareByPropertyCount(a, b) {
  return b.propertyCount - a.propertyCount || a.mask - b.mask || a.stableIndex - b.stableIndex;
}

function compareByMask(a, b) {
  return a.mask - b.mask || b.propertyCount - a.propertyCount || a.stableIndex - b.stableIndex;
}

function compareByOnes(a, b) {
  return b.ones - a.ones || b.propertyCount - a.propertyCount || a.stableIndex - b.stableIndex;
}

function compareById(a, b) {
  return a.stableIndex - b.stableIndex;
}

const SORTERS = {
  "propertyCount-desc": compareByPropertyCount,
  "propertyCount-asc": (a, b) => -compareByPropertyCount(a, b),
  "mask-asc": compareByMask,
  "ones-desc": compareByOnes,
  "id-asc": compareById,
};

export function ruleMatchesSelection(rule, selectedBits) {
  if (!selectedBits || selectedBits.length === 0) {
    return true;
  }
  return selectedBits.every((bit) => hasMaskBit(rule.mask, bit));
}

export function ruleMatchesSearch(rule, searchTerm, properties = null) {
  if (!searchTerm) {
    return true;
  }
  const needle = searchTerm.trim().toLowerCase();
  if (!needle) {
    return true;
  }
  const stableId = String(rule.stableId ?? "").toLowerCase();
  const stableIdShort = String(rule.stableIdShort ?? "").toLowerCase();
  const maskHex = String(rule.maskHex ?? "").toLowerCase();
  const lutHex = String(rule.lutHex ?? "").toLowerCase();
  const searchText = String(rule.searchText ?? "").toLowerCase();
  if (searchText && searchText.includes(needle)) {
    return true;
  }
  if (
    String(rule.id).includes(needle) ||
    String(rule.legacyIndex).includes(needle) ||
    String(rule.stableIndex).includes(needle) ||
    `rule ${rule.id}`.includes(needle) ||
    `legacy ${rule.legacyIndex}`.includes(needle) ||
    `stable ${rule.stableIndex}`.includes(needle) ||
    `sid ${stableId}`.includes(needle) ||
    `sid ${stableIdShort}`.includes(needle) ||
    stableId.includes(needle) ||
    stableIdShort.includes(needle) ||
    maskHex.includes(needle) ||
    String(rule.mask).includes(needle) ||
    lutHex.includes(needle)
  ) {
    return true;
  }
  if (!properties) {
    return false;
  }
  for (const property of properties) {
    if ((rule.mask & (1 << property.bit)) === 0) {
      continue;
    }
    if (
      property.label.toLowerCase().includes(needle) ||
      property.machineName.toLowerCase().includes(needle) ||
      property.category.toLowerCase().includes(needle)
    ) {
      return true;
    }
  }
  return false;
}

export function normalizeSidQuery(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/^sid:/, "")
    .replace(/[^0-9a-f]/g, "");
}

export function findRuleBySid(rules, sidQuery) {
  const prefix = normalizeSidQuery(sidQuery);
  if (!prefix) {
    return { status: "empty", rule: null, prefix };
  }
  const matches = rules.filter((rule) => rule.stableId.toLowerCase().startsWith(prefix));
  if (matches.length === 0) {
    return { status: "missing", rule: null, prefix };
  }
  if (matches.length > 1) {
    return { status: "ambiguous", rule: null, prefix };
  }
  return { status: "ok", rule: matches[0], prefix };
}

export function filterRules(rules, options) {
  const selectedBits = options.selectedBits ?? [];
  const searchTerm = (options.searchTerm ?? "").trim().toLowerCase();
  const properties = options.properties ?? null;
  const sortKey = options.sortKey ?? "propertyCount-desc";
  const sorter = SORTERS[sortKey] ?? SORTERS["propertyCount-desc"];
  return rules
    .filter(
      (rule) =>
        ruleMatchesSelection(rule, selectedBits) && ruleMatchesSearch(rule, searchTerm, properties),
    )
    .sort(sorter);
}

export function paginateRules(rules, pageSize, pageIndex) {
  const totalPages = Math.max(1, Math.ceil(rules.length / pageSize));
  const safePageIndex = Math.min(Math.max(pageIndex, 0), totalPages - 1);
  const start = safePageIndex * pageSize;
  return {
    totalPages,
    pageIndex: safePageIndex,
    items: rules.slice(start, start + pageSize),
  };
}
import { hasMaskBit } from "./lut.js";
