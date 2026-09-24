import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#123047",
        muted: "#617381",
        paper: "#f7f8f5",
        teal: {
          50: "#edfbf8",
          100: "#d5f5ee",
          500: "#159c8c",
          600: "#0f766e",
          700: "#0b5d58",
        },
        amber: {
          50: "#fff8e8",
          100: "#ffedbd",
          500: "#d49a24",
          600: "#b7791f",
        },
        coral: {
          50: "#fff1ed",
          500: "#dc6b5c",
          600: "#c45143",
        },
      },
      boxShadow: {
        card: "0 18px 50px rgba(18, 48, 71, 0.08)",
        soft: "0 8px 24px rgba(18, 48, 71, 0.06)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
