import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig(({ mode }) => {
  const basePath =
    process.env.VITE_BASE_PATH ||
    (mode === "production" ? "/rvc-webgui-fork/" : "/");

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
