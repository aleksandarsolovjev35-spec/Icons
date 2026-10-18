/* global Hooks, game, foundry, CONFIG, ui, Dialog, renderTemplate, MutationObserver, FormApplication */

import { resolveIcon } from "./matcher.js";

/**
 * Loot Icons for D&D 5e
 *
 * A template icon library: a small set of archetype icons is matched against
 * item names, so a hundred different "Potion of Healing" style items share
 * one readable silhouette instead of each needing bespoke art.
 *
 * The module does three things:
 *   1. ships the icons (icons/**.webp, transparent, 200x200)
 *   2. assigns them automatically to items (exact -> keyword -> type)
 *   3. decorates items with a runtime rarity highlight (CSS only, no extra art)
 */

const MODULE_ID = "loot-icons-dnd5e";

/** dnd5e ships only per-type placeholders, so this is the sensible last resort. */
const SYSTEM_FALLBACK = "systems/dnd5e/icons/svg/items/loot.svg";
const MYSTERY_MAN = "icons/svg/mystery-man.svg";

/** Paths the system itself uses, which we are allowed to replace. */
const SYSTEM_DEFAULTS = [
  "systems/dnd5e/icons/svg/items/",
  "systems/dnd5e/icons/svg/objects/",
  "icons/svg/mystery-man.svg"
];

/** rarity ids used by dnd5e, mapped to the colour used by the highlight */
const RARITY_COLOURS = {
  "common": null,
  "uncommon": "#6fcf5b",
  "rare": "#4aa3ff",
  "very-rare": "#b47cff",
  "legendary": "#ffb03a",
  "artifact": "#ff6b4a",
  "cursed": "#8b3fa8"
};

let library = null;
let libraryPromise = null;

/* ------------------------------------------------------------------ utils */

/** true when the current image is a system placeholder we may overwrite */
function isReplaceable(img) {
  if (!img) return true;
  if (img.startsWith(`modules/${MODULE_ID}/icons/`)) return true;
  if (img === MYSTERY_MAN) return true;
  return SYSTEM_DEFAULTS.some(prefix => img.startsWith(prefix));
}

/** public path of an icon id */
function iconPath(id) {
  return `modules/${MODULE_ID}/icons/${id}.webp`;
}

/* ----------------------------------------------------------------- library */

async function loadLibrary() {
  if (library) return library;
  if (!libraryPromise) {
    const get = path =>
      fetch(`modules/${MODULE_ID}/${path}`).then(response => {
        if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
        return response.json();
      });

    libraryPromise = Promise.all([get("data/icons.json"), get("data/rules.json")])
      .then(([icons, rules]) => {
        library = { icons: icons.icons ?? {}, categories: icons.categories ?? {}, rules: rules ?? {} };
        return library;
      })
      .catch(error => {
        console.error(`${MODULE_ID} | could not load the icon library`, error);
        library = { icons: {}, categories: {}, rules: { exact: {}, keywords: [], type: {} } };
        return library;
      });
  }
  return libraryPromise;
}

/* ------------------------------------------------------------- assignment */

/** assign an icon to a single item document */
async function assignToItem(item) {
  if (!item) return;
  if (!isReplaceable(item.img)) return;

  const data = await loadLibrary();
  const id = resolveIcon(item, data);
  if (!id) return;

  const path = iconPath(id);
  if (item.img === path) return;
  await item.update({ img: path });
}

/** assign icons to every item of an actor */
async function assignToActor(actor) {
  if (!actor) return 0;
  let count = 0;
  for (const item of actor.items ?? []) {
    const before = item.img;
    await assignToItem(item);
    if (item.img !== before) count += 1;
  }
  return count;
}

/** group the library by category, ready for the picker */
function libraryCategories() {
  const categories = new Map();
  for (const [id, path] of Object.entries(library?.icons ?? {})) {
    const category = id.includes("/") ? id.split("/")[0] : "misc";
    if (!categories.has(category)) categories.set(category, []);
    categories.get(category).push({
      id,
      path: `modules/${MODULE_ID}/${path}`,
      label: id.split("/").pop()
    });
  }
  return [...categories.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([id, icons]) => ({ id, label: id, icons }));
}

/**
 * Open the archetype picker for a document.
 * @param {Document} document  the item to change
 */
async function pickIcon(document) {
  if (!document) return;
  await loadLibrary();

  const content = await renderTemplate(
    `modules/${MODULE_ID}/templates/icon-picker.hbs`,
    { categories: libraryCategories() }
  );

  const dialog = new Dialog({
    title: game.i18n.localize("LOOTICONS.Preview"),
    content: `<div class="loot-picker-body">${content}</div>`,
    classes: ["loot-picker-dialog"],
    buttons: {
      cancel: {
        label: game.i18n.localize("Cancel"),
        callback: () => null
      }
    },
    default: "cancel"
  });

  dialog.render(true);
  const root = dialog.element?.[0] ?? dialog.element;
  root?.addEventListener("click", async event => {
    const entry = event.target.closest(".loot-picker-entry");
    if (!entry) return;
    await document.update({ img: `modules/${MODULE_ID}/icons/${entry.dataset.icon}.webp` });
    dialog.close();
  });
}

/* -------------------------------------------------------- rarity highlight */

/** read the rarity of an item in a way that survives both dnd5e generations */
function rarityOf(item) {
  const raw = item?.system?.rarity;
  if (raw === null || raw === undefined) return null;
  if (typeof raw === "number") {
    // 2024 style numeric rarity
    return ["common", "uncommon", "rare", "very-rare", "legendary", "artifact"][raw] ?? null;
  }
  const key = String(raw).toLowerCase().replace(/\s+/g, "-");
  if (!key || key === "none") return null;
  return key;
}

/**
 * Tag every rendered item row with its rarity so the stylesheet can tint it.
 * @param {HTMLElement} root   the sheet element
 * @param {object|null} source the actor or item the rows belong to
 */
function decorateItems(root, source) {
  if (!root) return;
  const rows = root.querySelectorAll(".item[data-item-id]");
  for (const row of rows) {
    const id = row.dataset.itemId;
    let item = null;
    if (source?.items?.get) item = source.items.get(id);
    else if (source?.id === id) item = source;
    const rarity = rarityOf(item);
    if (rarity && RARITY_COLOURS[rarity] !== undefined) {
      row.dataset.rarity = rarity;
    } else {
      delete row.dataset.rarity;
    }
  }
}

/** keep an observer alive per rendered sheet */
const observers = new WeakMap();

function watchSheet(app, html) {
  if (!game.settings.get(MODULE_ID, "rarityHighlight")) return;
  const root = html?.[0] ?? html;
  if (!root) return;

  const paint = () => decorateItems(root, app.document);
  paint();

  if (observers.has(root)) observers.get(root).disconnect();
  const observer = new MutationObserver(() => paint());
  observer.observe(root, { childList: true, subtree: true });
  observers.set(root, observer);
}

/** the DOM element of a rendered sheet, across Foundry versions */
function sheetElement(app, html) {
  return html?.[0] ?? app?.element?.[0] ?? app?.element ?? null;
}

/** find (or create) the button row of a sheet header */
function headerActions(root) {
  if (!root) return null;
  let actions = root.querySelector(".sheet-header .header-actions");
  if (!actions) {
    const header = root.querySelector(".sheet-header");
    if (!header) return null;
    actions = document.createElement("div");
    actions.className = "header-actions";
    header.appendChild(actions);
  }
  return actions;
}

/** add a small icon button to a sheet header */
function addHeaderButton(root, { icon, title, onClick }) {
  const actions = headerActions(root);
  if (!actions) return;
  if (actions.querySelector(`.loot-icons-button[data-icon="${icon}"]`)) return;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "loot-icons-button";
  button.dataset.icon = icon;
  button.title = title;
  button.setAttribute("aria-label", title);
  button.innerHTML = `<i class="fas ${icon}"></i>`;
  button.addEventListener("click", onClick);
  actions.prepend(button);
}

/* --------------------------------------------------------------- settings */

function registerSettings() {
  game.settings.register(MODULE_ID, "autoAssign", {
    name: "Automatic assignment",
    hint: "Assign an icon to newly created items and to items whose name changed.",
    scope: "world",
    config: true,
    type: Boolean,
    default: true
  });

  game.settings.register(MODULE_ID, "rarityHighlight", {
    name: "Rarity highlight",
    hint: "Tint and glow item icons by rarity. Purely cosmetic, no extra artwork.",
    scope: "client",
    config: true,
    type: Boolean,
    default: true
  });
}

/* ------------------------------------------------------------- public API */

/** API surface available to macros and other modules */
globalThis.LootIcons = {
  MODULE_ID,
  SYSTEM_FALLBACK,
  RARITY_COLOURS,
  get library() {
    return library;
  },
  load: loadLibrary,
  iconPath,
  resolveIcon: item => resolveIcon(item, library ?? { icons: {}, rules: {} }),
  assignToItem,
  assignToActor,
  pickIcon,
  categories: libraryCategories,

  /**
   * Batch assignment.
   * @param {object}  options
   * @param {boolean} options.compendiums  also walk every compendium
   */
  async assignAll({ compendiums = false } = {}) {
    let count = 0;
    for (const actor of game.actors ?? []) {
      for (const item of actor.items ?? []) {
        const before = item.img;
        await assignToItem(item);
        if (item.img !== before) count += 1;
      }
    }
    for (const item of game.items ?? []) {
      const before = item.img;
      await assignToItem(item);
      if (item.img !== before) count += 1;
    }

    if (compendiums) {
      for (const pack of game.packs ?? []) {
        if (!pack.metadata?.type || pack.metadata.type !== "Item") continue;
        const documents = pack.documentName === "Item" ? await pack.getDocuments() : [];
        for (const item of documents) {
          const before = item.img;
          await assignToItem(item);
          if (item.img !== before) count += 1;
        }
      }
    }
    return count;
  },

  /** list the library, optionally filtered by category */
  list(category) {
    const icons = library?.icons ?? {};
    return Object.keys(icons)
      .filter(id => !category || id.startsWith(`${category}/`))
      .map(id => ({ id, path: iconPath(id) }));
  }
};

/* --------------------------------------------------------------- lifecycle */

Hooks.once("init", () => {
  registerSettings();
  loadLibrary();
});

Hooks.on("createItem", (document, options, userId) => {
  if (!game.settings.get(MODULE_ID, "autoAssign")) return;
  if (userId !== game.user?.id) return;
  assignToItem(document);
});

Hooks.on("updateItem", (document, changes, options, userId) => {
  if (!game.settings.get(MODULE_ID, "autoAssign")) return;
  if (userId !== game.user?.id) return;
  if (!changes?.name) return; // only react to renames
  assignToItem(document);
});

Hooks.on("createActor", (actor, options, userId) => {
  if (!game.settings.get(MODULE_ID, "autoAssign")) return;
  if (userId !== game.user?.id) return;
  assignToActor(actor);
});

Hooks.on("renderActorSheet", (app, html) => {
  watchSheet(app, html);
  const root = sheetElement(app, html);
  addHeaderButton(root, {
    icon: "fa-wand-magic-sparkles",
    title: game.i18n.localize("LOOTICONS.AssignAll"),
    onClick: async () => {
      const count = await assignToActor(app.document);
      ui.notifications?.info(game.i18n.format("LOOTICONS.Assigned", { count }));
    }
  });
});

Hooks.on("renderItemSheet", (app, html) => {
  watchSheet(app, html);
  const root = sheetElement(app, html);
  addHeaderButton(root, {
    icon: "fa-images",
    title: game.i18n.localize("LOOTICONS.Preview"),
    onClick: () => pickIcon(app.document)
  });
});

Hooks.on("renderActorSheetV2", (app, html) => {
  watchSheet(app, html);
  const root = sheetElement(app, html);
  addHeaderButton(root, {
    icon: "fa-wand-magic-sparkles",
    title: game.i18n.localize("LOOTICONS.AssignAll"),
    onClick: async () => {
      const count = await assignToActor(app.document);
      ui.notifications?.info(game.i18n.format("LOOTICONS.Assigned", { count }));
    }
  });
});
