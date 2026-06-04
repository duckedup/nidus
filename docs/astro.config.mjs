// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// nidus docs — Astro + Starlight, deployed to GitHub Pages at a custom domain.
// Custom domain serves from the root, so `base` stays "/".
export default defineConfig({
  site: "https://nidus.duckedup.org",
  integrations: [
    starlight({
      title: "nidus",
      description:
        "A small, pure-Rust vector store for development and small-scale use. Brute-force cosine search over a single append-only directory — embed it as a library, with a standalone server planned. No FFI, no C, no SQL, no query engine.",
      logo: {
        // The nest mark — full-colour illustration, reads on light and dark.
        src: "./src/assets/nidus.svg",
        alt: "nidus",
      },
      // Code blocks wear Everforest — a warm, woodland palette that matches the
      // nest. Dark theme first, light second; Starlight switches with the page.
      expressiveCode: {
        themes: ["everforest-dark", "everforest-light"],
        styleOverrides: {
          borderRadius: "0.5rem",
          borderColor: "var(--sl-color-gray-5)",
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
            { label: "Search & filters", link: "/guides/search/" },
            { label: "Embedding in a host app", link: "/guides/integrating/" },
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
