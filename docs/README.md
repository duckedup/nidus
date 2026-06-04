# nidus docs

The documentation site for [nidus](https://github.com/duckedup/nidus), built
with [Astro](https://astro.build) + [Starlight](https://starlight.astro.build)
and deployed to GitHub Pages at **https://nidus.duckedup.org**.

## Local development

From the repo root:

```bash
just docs          # dev server with live reload (http://localhost:4321)
just docs-build    # production build → docs/dist/
just docs-preview  # preview the production build
```

Or directly with [Bun](https://bun.sh):

```bash
cd docs
bun install
bun run dev
```

## Structure

```
docs/
├── astro.config.mjs          # site config, sidebar, Everforest code theme
├── src/
│   ├── content/docs/         # the docs pages (Markdown / MDX)
│   ├── styles/nest.css       # the "nest" theme
│   └── assets/nidus.svg      # the nest mark
└── public/
    ├── CNAME                 # custom domain
    └── favicon.svg
```

## Theme

The "nest" theme (`src/styles/nest.css`) maps the colours of the nidus mark — a
bird's nest with three painted eggs — onto Starlight's design tokens: bark browns
for the dark ground, warm parchment for light, a single teal accent (the painted
egg), and gold (the nest's flowers) reserved as the one flourish. Code blocks use
the [Everforest](https://github.com/sainnhe/everforest) syntax theme — a warm,
woodland palette — in both modes. Fraunces (display) + Hanken Grotesk (text) +
JetBrains Mono (code).

## Deployment

Pushing to `main` with changes under `docs/**` triggers
`.github/workflows/docs.yml`, which builds the site and publishes it to GitHub
Pages.
