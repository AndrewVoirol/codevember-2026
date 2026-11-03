#!/usr/bin/env node
/**
 * Syncs custom `andrewvoirol.com` subdomains (`codevember-2026.andrewvoirol.com`
 * and `codevember-2026-day-XX.andrewvoirol.com`) across Vercel and Cloudflare DNS.
 *
 * Usage:
 *   npm run domains:sync               # syncs hub + all completed days
 *   npm run domains:sync -- --day 01   # syncs hub + Day 01 specifically
 *   npm run domains:sync -- --all      # pre-provisions all 30 day subdomains
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const ZONE_ID = 'd58dd4efb58da5dd897a8de67d82a62c'; // andrewvoirol.com
const VERCEL_PROJECT = 'codevember-2026';
const BASE_DOMAIN = 'andrewvoirol.com';

function getTargetDays() {
  const args = process.argv.slice(2);
  if (args.includes('--all')) {
    return Array.from({ length: 30 }, (_, i) => i + 1);
  }
  const dayIdx = args.indexOf('--day');
  if (dayIdx !== -1 && args[dayIdx + 1]) {
    return [Number.parseInt(args[dayIdx + 1], 10)];
  }
  // Default: parse completed days from src/core/manifest.ts
  const manifest = fs.readFileSync(path.resolve(ROOT_DIR, 'src/core/manifest.ts'), 'utf8');
  const days = [];
  const regex = /createDay\(\s*(\d+)\s*,\s*'[^']+'\s*,\s*'complete'/g;
  let m;
  while ((m = regex.exec(manifest)) !== null) {
    const d = Number.parseInt(m[1], 10);
    if (d >= 1 && d <= 30) days.push(d);
  }
  return days;
}

async function ensureCloudflareCname(sub, token) {
  if (!token) {
    console.warn(`  [Cloudflare] Skipped ${sub}.${BASE_DOMAIN} (CLOUDFLARE_API_TOKEN not set)`);
    return;
  }
  const fqdn = `${sub}.${BASE_DOMAIN}`;
  const listRes = await fetch(
    `https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?type=CNAME&name=${fqdn}`,
    { headers: { Authorization: `Bearer ${token}` } }
  );
  const listJson = await listRes.json();
  if (!listJson.success) {
    console.warn(`  [Cloudflare] Auth/API error for ${fqdn}:`, JSON.stringify(listJson.errors));
    return;
  }
  if (listJson.result && listJson.result.length > 0) {
    console.log(`  [Cloudflare] CNAME already exists: ${fqdn} -> ${listJson.result[0].content}`);
    return;
  }

  const createRes = await fetch(
    `https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        type: 'CNAME',
        name: sub,
        content: 'cname.vercel-dns.com',
        proxied: false,
        ttl: 1,
      }),
    }
  );
  const createJson = await createRes.json();
  if (createJson.success) {
    console.log(`  [Cloudflare] Created CNAME: ${fqdn} -> cname.vercel-dns.com`);
  } else {
    console.warn(`  [Cloudflare] Failed creating ${fqdn}:`, JSON.stringify(createJson.errors));
  }
}

function ensureVercelDomain(fqdn) {
  try {
    execFileSync('vercel', ['domains', 'add', fqdn, VERCEL_PROJECT], {
      cwd: ROOT_DIR,
      env: { ...process.env, NO_UPDATE_NOTIFIER: '1' },
      stdio: 'pipe',
    });
    console.log(`  [Vercel] Attached domain: ${fqdn}`);
  } catch (err) {
    const out = String(err.stderr || err.stdout || err.message);
    if (/already assigned|already exists/i.test(out)) {
      console.log(`  [Vercel] Domain already attached: ${fqdn}`);
    } else {
      console.warn(`  [Vercel] Warning on ${fqdn}: ${out.trim()}`);
    }
  }
}

async function main() {
  const token = process.env.CLOUDFLARE_API_TOKEN;
  const days = getTargetDays();
  const subdomains = [
    'codevember-2026',
    ...days.map((d) => `codevember-2026-day-${String(d).padStart(2, '0')}`),
  ];

  for (const sub of subdomains) {
    const fqdn = `${sub}.${BASE_DOMAIN}`;
    console.log(`Syncing ${fqdn}...`);
    ensureVercelDomain(fqdn);
    await ensureCloudflareCname(sub, token);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
