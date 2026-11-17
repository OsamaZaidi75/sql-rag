import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev only: the API runs on :8000, the built app is served by FastAPI itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
