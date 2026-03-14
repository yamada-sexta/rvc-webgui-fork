import { defineConfig, loadEnv } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const basePath = env.VITE_BASE_PATH || "/";

  return {
    base: basePath,
    plugins: [solidPlugin()],
    server: {
      port: 3000,
    },
    build: {
      target: "esnext",
    },
  };
});
