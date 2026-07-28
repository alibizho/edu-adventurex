declare module "*.css";

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_DATA_MODE?: "backend" | "mock";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
