#!/usr/bin/env node
/**
 * Syncs custom `andrewvoirol.com` subdomains (`codevember-2026.andrewvoirol.com`
 * and `codevember-2026-day-XX.andrewvoirol.com`) across Vercel and Cloudflare DNS
 * via direct REST APIs (completes 31 domains in ~2 seconds with zero CLI spawns).
 *
 * Usage:
 *   npm run domains:sync               # syncs hub + all completed days
 *   npm run domains:sync -- --day 01   # syncs hub + Day 01 specifically
 *   npm run domains:sync -- --all      # pre-provisions all 30 day subdomains
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const ZONE_ID = 'd58dd4efb58da5dd897a8de67d82a62c'; // andrewvoirol.com
const VERCEL_PROJECT_ID = 'prj_phmPwslPBCW72F6PmGRnuHCcgq36';
const VERCEL_TEAM_ID = 'team_kolNPWOwylfRLVW2zjkXO3Ll';
const BASE_DOMAIN = 'andrewvoirol.com';

function resolveVercelToken() {
  if (process.env.VERCEL_TOKEN) return process.env.VERCEL_TOKEN;
  const candidates = [
    path.join(os.homedir(), 'Library/Application Support/com.vercel.cli/auth.json'),
    path.join(os.homedir(), '.local/share/com.vercel.cli/auth.json'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) {
      try {
        const json = JSON.parse(fs.readFileSync(p, 'utf8'));
        if (json.token) return json.token;
      } catch {
        // ignore
      }
    }
  }
  return null;
}

function getTargetDays() {
  const args = process.argv.slice(2);
  if (args.includes('--all')) {
    return Array.from({ length: 30 }, (_, i) => i + 1);
  }
  const dayIdx = args.indexOf('--day');
  if (dayIdx !== -1 && args[dayIdx + 1]) {
    return [Number.parseInt(args[dayIdx + 1], 10)];
  }
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

async function fetchExistingCloudflareCnames(cfToken) {
  if (!cfToken) return new Set();
  const res = await fetch(
    `https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records?type=CNAME&per_page=100`,
    { headers: { Authorization: `Bearer ${cfToken}` } }
  );
  const json = await res.json();
  if (!json.success) {
    console.warn('[Cloudflare] Warning listing CNAMEs:', JSON.stringify(json.errors));
    return new Set();
  }
  return new Set((json.result || []).map((r) => r.name));
}

async function ensureCloudflareCname(sub, cfToken, existingSet) {
  const fqdn = `${sub}.${BASE_DOMAIN}`;
  if (!cfToken) {
    console.warn(`  [Cloudflare] Skipped ${fqdn} (CLOUDFLARE_API_TOKEN not set)`);
    return;
  }
  if (existingSet.has(fqdn)) {
    return;
  }
  const createRes = await fetch(
    `https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${cfToken}`,
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

async function ensureVercelDomainApi(fqdn, vercelToken) {
  if (!vercelToken) {
    console.warn(`  [Vercel] Skipped ${fqdn} (no Vercel token found)`);
    return;
  }
  const res = await fetch(
    `https://api.vercel.com/v10/projects/${VERCEL_PROJECT_ID}/domains?teamId=${VERCEL_TEAM_ID}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${vercelToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: fqdn }),
    }
  );
  const json = await res.json();
  if (res.ok || json?.error?.code === 'domain_already_in_use' || json?.error?.code === 'domain_already_exists') {
    return true;
  }
  console.warn(`  [Vercel] Warning on ${fqdn}:`, JSON.stringify(json.error || json));
  return false;
}

async function main() {
  const cfToken = process.env.CLOUDFLARE_API_TOKEN;
  const vercelToken = resolveVercelToken();
  const days = getTargetDays();
  const subdomains = [
    'codevember-2026',
    ...days.map((d) => `codevember-2026-day-${String(d).padStart(2, '0')}`),
  ];

  const existingCf = await fetchExistingCloudflareCnames(cfToken);

  // Process in parallel batches of 6
  const batchSize = 6;
  for (let i = 0; i < subdomains.length; i += batchSize) {
    const slice = subdomains.slice(i, i + batchSize);
    await Promise.all(
      slice.map(async (sub) => {
        const fqdn = `${sub}.${BASE_DOMAIN}`;
        await Promise.all([
          ensureVercelDomainApi(fqdn, vercelToken),
          ensureCloudflareCname(sub, cfToken, existingCf),
        ]);
      })
    );
  }
  console.log(`Synced ${subdomains.length} domain(s) across Vercel & Cloudflare DNS.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
