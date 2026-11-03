import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function discoverMpaInputs(rootDir: string): Record<string, string> {
  const inputs: Record<string, string> = {
    main: path.resolve(rootDir, 'index.html'),
  };

  if (!fs.existsSync(rootDir)) {
    return inputs;
  }

  const entries = fs.readdirSync(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (/^[0-3][0-9]-[a-z0-9-]+$/.test(entry.name)) {
      const htmlPath = path.resolve(rootDir, entry.name, 'index.html');
      if (fs.existsSync(htmlPath)) {
        inputs[entry.name] = htmlPath;
      }
    }
  }

  return inputs;
}

export default defineConfig({
  appType: 'mpa',
  build: {
    target: 'esnext',
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      input: discoverMpaInputs(__dirname),
    },
  },
});
