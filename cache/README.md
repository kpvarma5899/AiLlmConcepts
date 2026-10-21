# Canvas to pages

Generates the GitHub Pages site in `docs/` from each question folder.

```bash
node cache/canvas-to-pages.mjs
```

Run that from the repository root. A question is `questions/<slug>/`:

- `README.md` — the problem. The first line after the title is the page link. The renderer skips that line.
- `page.json` — number and kicker. Stats come from the blank-header table in the note.
- `canvas.tsx` — the Cursor canvas, when this question was written from one. The renderer does not read it.
- `src/` — the types. The renderer does not read it.

A note can place a widget with an HTML comment:

- `<!-- widget:recall-ef -->`
- `<!-- widget:latency-ef -->`
- `<!-- widget:storage-50m -->`

`docs/` is generated. Change the note or this cache, then run the script again. Do not edit `docs/` by hand.
