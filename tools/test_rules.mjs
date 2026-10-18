/**
 * Unit tests for the icon matching rules.
 *
 *   node tools/test_rules.mjs
 *
 * Uses the very same matcher the Foundry module imports, so a broken rule in
 * data/rules.json is caught without booting Foundry.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { resolveIcon, normalise } from "../scripts/matcher.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const icons = JSON.parse(readFileSync(join(ROOT, "data", "icons.json"), "utf8"));
const rules = JSON.parse(readFileSync(join(ROOT, "data", "rules.json"), "utf8"));
const data = { icons: icons.icons, rules };

/** [name, expected icon id, item type] — a null expectation means "no match" */
const CASES = [
  // exact rules
  ["Potion of Healing", "consumables/flask-potion"],
  ["potion of greater healing", "consumables/flask-potion"],
  ["Antidote", "consumables/vial"],
  ["Bag of Holding", "containers/sack"],
  ["Iron Ingot", "treasure/ingot"],
  ["Gold", "treasure/coin-stack"],

  // keyword rules
  ["Elixir of Speed", "consumables/flask-potion"],
  ["Flask of Wine", "consumables/bottle"],
  ["Bottle of Fine Wine", "consumables/bottle"],
  ["Bottle of Poison", "consumables/vial"],
  ["Vial of Antidote", "consumables/vial"],
  ["Lamp Oil", "consumables/vial"],
  ["Scroll of Fireball", "consumables/scroll"],
  ["Map of the Underdark", "consumables/scroll"],
  ["Spellbook", "adventuring/book"],
  ["Grimoire of Madness", "adventuring/book"],
  ["Everlasting Torch", "adventuring/torch"],
  ["Oil Lantern", "adventuring/lantern"],
  ["Rope (50 feet)", "adventuring/rope"],
  ["Hempen Rope", "adventuring/rope"],
  ["Adventurer's Pack", "containers/backpack"],
  ["Leather Backpack", "containers/backpack"],
  ["Treasure Chest", "containers/chest"],
  ["Strongbox", "containers/chest"],
  ["Bag of Grain", "containers/sack"],
  ["Coin Pouch", "treasure/coin-stack"],
  ["Gold Coins", "treasure/coin-stack"],
  ["Silver Ingot", "treasure/ingot"],
  ["Signet Ring", "treasure/ring"],
  ["Ruby Ring", "treasure/ring"],
  ["Raw Amethyst", "treasure/gem"],
  ["Iron Ore", "materials/ore"],
  ["Glowing Mushroom", "nature/mushroom"],
  ["Loaf of Bread", "food/bread"],
  ["Trail Rations", "food/bread"],

  // type fallback
  ["Mystery Item", "consumables/flask-potion", "consumable"],
  ["Mystery Crate", "containers/sack", "container"],

  // must not match anything
  ["", null, "weapon"],
  ["Zzzz Qqqq", null, "weapon"]
];

let failed = 0;
let skipped = 0;
for (const [name, expected, type] of CASES) {
  if (expected && !icons.icons[expected]) {
    skipped += 1;
    console.log(`SKIP  ${JSON.stringify(name)} - ${expected} is not in the library yet`);
    continue;
  }
  const actual = resolveIcon({ name, type }, data);
  const ok = actual === expected;
  if (!ok) {
    failed += 1;
    console.error(`FAIL  ${JSON.stringify(name)}\n      expected ${expected}, got ${actual}`);
  }
}

// a rule that points at a missing icon must not win
const broken = { icons: {}, rules: { exact: { "x": "nope" }, keywords: [{ icon: "nope", words: ["x"] }] } };
if (resolveIcon({ name: "x" }, broken) !== null) {
  failed += 1;
  console.error("FAIL  rules pointing at missing icons must be ignored");
}

// normalisation
const norm = normalise("  Potion of   Healing!  ");
if (norm !== "potion of healing") {
  failed += 1;
  console.error(`FAIL  normalise -> ${norm}`);
}

const total = CASES.length + 2;
console.log(`${total - failed - skipped}/${total} checks passed${skipped ? ` (${skipped} skipped)` : ""}`);
process.exit(failed ? 1 : 0);
