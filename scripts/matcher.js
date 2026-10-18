/**
 * Pure matching logic for Loot Icons for D&D 5e.
 *
 * Kept free of Foundry globals so it can be unit tested with plain node
 * (see tools/test_rules.mjs).
 */

/** lowercase, strip punctuation, collapse whitespace */
export function normalise(text) {
  return String(text ?? "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9а-яё]+/gi, " ")
    .trim();
}

/** regex that matches a phrase on token boundaries */
export function phraseRegex(phrase) {
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|\\s)${escaped}(\\s|$)`, "i");
}

/**
 * Resolve an icon id for an item.
 *
 * Matching order: exact name, then keyword rules (most matches wins),
 * then the per-type fallback.
 *
 * @param {{name?: string, type?: string}} item
 * @param {{icons: object, rules: object}} data
 * @returns {string|null} icon id such as "consumables/flask-potion"
 */
export function resolveIcon(item, data) {
  const name = normalise(item?.name);
  if (!name) return null;

  const icons = data?.icons ?? {};
  const rules = data?.rules ?? {};

  // 1. exact name match
  const exact = rules.exact?.[name];
  if (exact && icons[exact]) return exact;

  // 2. keyword rules. A plain word scores 1, a `strong` word scores 3, so a
  //    specific content word ("poison") outweighs a generic one ("bottle").
  let best = null;
  let bestScore = 0;
  for (const rule of rules.keywords ?? []) {
    if (!icons[rule.icon]) continue;
    let score = 0;
    for (const word of rule.words ?? []) {
      if (phraseRegex(normalise(word)).test(name)) score += 1;
    }
    for (const word of rule.strong ?? []) {
      if (phraseRegex(normalise(word)).test(name)) score += 3;
    }
    if (score > bestScore) {
      bestScore = score;
      best = rule.icon;
    }
  }
  if (best) return best;

  // 3. per-type fallback
  const type = item?.type;
  if (type && rules.type?.[type] && icons[rules.type[type]]) {
    return rules.type[type];
  }

  return null;
}
