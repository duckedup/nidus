// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// nidus docs — Astro + Starlight, deployed to GitHub Pages at a custom domain.
// Custom domain serves from the root, so `base` stays "/".
export default defineConfig({
  site: "https://nidus.duckedup.org",
  // The combined "backends" guide was split into Storage + Memory pages.
  redirects: {
    "/guides/backends/": "/guides/storage-backends/",
  },
  integrations: [
    starlight({
      title: "nidus",
      description:
        "A small, pure-Rust vector store for development and small-scale use. Brute-force cosine search over a single append-only directory — no FFI, no C, no SQL, no query engine.",
      logo: {
        // The nest mark — full-colour illustration, reads on light and dark.
        src: "./src/assets/nidus.svg",
        alt: "nidus",
      },
      components: {
        // Custom splash hero: the nest in an ember glow, overlaid with a live
        // nearest-neighbour weave. See src/components/Hero.astro.
        Hero: "./src/components/Hero.astro",
        // One sidebar only: a custom left nav with icon-led, collapsible
        // sections. PageSidebar is emptied to drop the right-hand TOC.
        Sidebar: "./src/components/Sidebar.astro",
        PageSidebar: "./src/components/PageSidebar.astro",
        // Adds Astro view transitions for flash-free navigation.
        Head: "./src/components/Head.astro",
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
      ],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Introduction", link: "/" },
            { label: "Getting started", link: "/getting-started/" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "How it works", link: "/guides/how-it-works/" },
            { label: "Storage & durability", link: "/guides/storage/" },
            { label: "Storage backends", link: "/guides/storage-backends/" },
            { label: "Memory stores", link: "/guides/memory-stores/" },
            { label: "Search & filters", link: "/guides/search/" },
            { label: "Remember & recall", link: "/guides/remember-and-recall/" },
            { label: "Embedding in a host app", link: "/guides/integrating/" },
            { label: "Command line", link: "/guides/cli-and-server/" },
            { label: "HTTP server", link: "/guides/http-server/" },
            { label: "Kubernetes (Helm)", link: "/guides/kubernetes/" },
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
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "API", link: "/reference/api/" },
            { label: "Configuration", link: "/reference/configuration/" },
            { label: "Performance", link: "/reference/performance/" },
          ],
        },
      ],
    }),
  ],
});
