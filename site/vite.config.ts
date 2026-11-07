import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Served from https://arjunkshah12345-hash.github.io/kimi-k3-forensics/
export default defineConfig(({ mode }) => ({
  base: "/kimi-k3-forensics/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  // dither-kit reads process.env.NODE_ENV for dev-only warnings
  define: { "process.env.NODE_ENV": JSON.stringify(mode === "production" ? "production" : "development") },
}))
