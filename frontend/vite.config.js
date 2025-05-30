import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": "/src",
    },
    extensions: [".js", ".jsx"],  // ✅ Ensure Vite recognizes JSX files
  },
  server: {
    port: 5173,
  },
});