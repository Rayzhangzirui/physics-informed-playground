# Playground build pipeline

This project uses **npm** scripts for building and watching the BILO Playground (TypeScript + HTML/CSS).

## Prerequisites

- Node.js (v16+)
- npm

## Build commands

| Command | Description |
|--------|-------------|
| `npm run prep` | Copy static assets (HTML, CSS, lib.js) into `dist/` |
| `npm run build-bilo` | Bundle BILO playground TS → `dist/bilo.js` |
| `npm run build-css` | Concatenate CSS → `dist/bundle.css` |
| `npm run build-html` | Copy/build HTML into `dist/` |
| **`npm run build`** | Run prep + build-bilo + build-css + build-html (full build) |

## Watch (development)

| Command | Description |
|--------|-------------|
| `npm run watch-bilo` | Watch BILO sources; rebuild `dist/bilo.js` on change |
| `npm run watch-js` | Watch main playground; rebuild `dist/bundle.js` on change |
| `npm run watch-css` | Rebuild CSS on change |
| `npm run watch-html` | Rebuild HTML on change |
| **`npm run watch`** | Run all watchers in parallel (prep + watch-bilo, watch-js, watch-css, watch-html) |

## Serve and watch together

```bash
npm run serve-watch
```

Starts the static server and all watchers so the app auto-rebuilds on file changes. Open the BILO page at the URL shown (e.g. `http://localhost:3000/bilo.html`).

## BILO-specific

- **Entry:** `src/bilo_playground.ts` → `dist/bilo.js`
- **Page:** `bilo.html` (and `bilo.css`) are copied to `dist/` by `prep`; load `dist/bilo.js` in `bilo.html`.
- **Tests:** `npm run test-bilo` — builds the test bundle and runs Node tests.
- **Snapshot for Python verification:** `npm run write-bilo-snapshot` — writes `bilo_np/ts_snapshot.json`. See `bilo_np/README_VERIFY_TS.md`.

## Clean

```bash
npm run clean
```

Removes the `dist/` directory.

---

**Note (2026):** The main JS bundle is minified with **terser** (not uglify-js). Old uglify-js only understands ES5; tsify outputs ES6+ (`let`/`const`), so the build was updated to use terser for the `build-js` step.
