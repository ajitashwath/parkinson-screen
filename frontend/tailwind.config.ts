import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#222f30",
        abyss: "#222f30",
        carbon: "#000000",
        muted: "#4d5757",
        graphite: "#585858",
        smoke: "#888888",
        ash: "#a0a0a0",
        silver: "#b8b8b8",
        paper: "#faf9f6",
        cream: "#faf9f6",
        linen: "#f1eee9",
        stone: "#d3cec6",
        hairline: "#dedbd6",
        violet: "#0007cb",
        lime: {
          50: "#f2f9e7",
          100: "#cef79e",
          200: "#b6e88a",
          500: "#cef79e",
          600: "#222f30",
          700: "#111111",
        },
        teal: {
          50: "#f2f9e7",
          100: "#cef79e",
          200: "#b6e88a",
          500: "#cef79e",
          600: "#222f30",
          700: "#111111",
        },
        amber: {
          50: "#f1eee9",
          100: "#dedbd6",
          200: "#d3cec6",
          500: "#a0a0a0",
          600: "#4d5757",
        },
        coral: {
          50: "#f1eee9",
          500: "#0007cb",
          600: "#0007cb",
        },
      },
      boxShadow: {
        card: "none",
        soft: "none",
      },
      maxWidth: {
        "7xl": "1200px",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
