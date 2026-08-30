// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// nidus docs — Astro + Starlight, deployed to GitHub Pages at a custom domain.
// Custom domain serves from the root, so `base` stays "/".
export default defineConfig({
  site: "https://nidus.duckedup.org",
  // The home page terminal's worker dynamically imports the wasm binding, so it is a
  // code-splitting build. Vite's default worker format is iife, which cannot split.
  vite: { worker: { format: "es" } },
  // The combined "backends" guide was split into Storage + Memory pages.
  redirects: {
    "/guides/backends/": "/guides/storage-backends/",
  },
  integrations: [
    starlight({
      title: "nidus",
      description:
        "One binary, one directory. A vector store you search from your shell, over HTTP, or as memory for an agent: semantic and keyword search, AST-aware code search, and MCP. Pure Rust: no FFI, no C, no SQL, no query engine.",
      logo: {
        // The nest mark — full-colour illustration, reads on light and dark.
        src: "./src/assets/nidus.svg",
        alt: "nidus",
      },
      components: {
        // One sidebar only: a custom left nav with icon-led, collapsible
        // sections. PageSidebar is emptied to drop the right-hand TOC.
        Sidebar: "./src/components/Sidebar.astro",
        PageSidebar: "./src/components/PageSidebar.astro",
        // Adds Astro view transitions for flash-free navigation.
        Head: "./src/components/Head.astro",
        // Persists the header (and with it Starlight's search) across those
        // transitions. See src/components/Header.astro.
        Header: "./src/components/Header.astro",
      },
      // Code blocks wear Everforest — a warm, woodland palette that matches the
      // nest. Dark theme first, light second; Starlight switches with the page.
      expressiveCode: {
        themes: ["everforest-dark", "everforest-light"],
        styleOverrides: {
          borderRadius: "0.6rem",
          borderColor: "var(--sl-color-hairline)",
        },
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/duckedup/nidus",
        },
      ],
      customCss: [
        "@fontsource-variable/fraunces",
        "@fontsource-variable/hanken-grotesk",
        "@fontsource/jetbrains-mono/400.css",
        "@fontsource/jetbrains-mono/500.css",
        "./src/styles/nest.css",
        // Shared with the landing page (src/pages/index.astro), which runs
        // outside Starlight and so cannot pull styles from nest.css.
        "./src/styles/terminal.css",
      ],
      sidebar: [
        {
          label: "Start here",
          items: [
            // "/" is the landing page (src/pages/index.astro), not a docs page.
            { label: "Home", link: "/" },
            { label: "Getting started", link: "/getting-started/" },
          ],
        },
        {
          label: "Core",
          items: [
            { label: "How it works", link: "/guides/how-it-works/" },
            { label: "Storage & durability", link: "/guides/storage/" },
            { label: "Vector search", link: "/guides/search/" },
            { label: "Full-text search (BM25)", link: "/guides/full-text-search/" },
            { label: "Hybrid search (RRF)", link: "/guides/hybrid-search/" },
            { label: "Filters & metadata", link: "/guides/filters/" },
            { label: "Reranking", link: "/guides/rerank/" },
            { label: "Remember & recall", link: "/guides/remember-and-recall/" },
          ],
        },
        {
          label: "Loading data",
          items: [
            { label: "Ingest a directory", link: "/guides/ingest/" },
          ],
        },
        {
          label: "Operating",
          items: [
            { label: "Command line", link: "/guides/cli-and-server/" },
            { label: "HTTP server", link: "/guides/http-server/" },
            { label: "Storage backends", link: "/guides/storage-backends/" },
            { label: "In-memory tier", link: "/guides/in-memory-tier/" },
            { label: "Blue/green reindexing", link: "/guides/blue-green-reindex/" },
            { label: "Running across a few boxes", link: "/guides/multi-box/" },
            { label: "Kubernetes (Helm)", link: "/guides/kubernetes/" },
          ],
        },
        {
          label: "Embedding nidus",
          items: [
            { label: "In a host app", link: "/guides/integrating/" },
            { label: "In the browser (wasm)", link: "/guides/wasm/" },
          ],
        },
        {
          label: "Also built in",
          items: [
            { label: "MCP", link: "/guides/mcp/" },
            { label: "Code search", link: "/guides/code/" },
            { label: "Automatic memory", link: "/guides/automatic-memory/" },
          ],
        },
        {
          label: "HTTP API",
          items: [
            { label: "Endpoint reference", link: "/reference/http-api/" },
          ],
        },
        {
          label: "SDKs",
          items: [
            { label: "JavaScript / TypeScript", link: "/sdks/javascript/" },
            { label: "Go", link: "/sdks/go/" },
            { label: "Python", link: "/sdks/python/" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "CLI", link: "/reference/cli/" },
            { label: "API", link: "/reference/api/" },
            { label: "Configuration", link: "/reference/configuration/" },
            { label: "Performance", link: "/reference/performance/" },
          ],
        },
      ],
    }),
  ],
});
