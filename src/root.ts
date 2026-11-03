import { getDayByNumber, getLatestActiveDay } from './core/manifest';

type DayModuleLoader = () => Promise<Record<string, unknown>>;

const dayModules = import.meta.glob('./days/*/main.ts') as Record<string, DayModuleLoader>;

function resolveRootTargetDay() {
  if (typeof window !== 'undefined') {
    const hostMatch = window.location.hostname.match(/day-(\d{1,2})\b/i);
    if (hostMatch) {
      const parsed = Number.parseInt(hostMatch[1], 10);
      if (parsed >= 0 && parsed <= 30) {
        return getDayByNumber(parsed);
      }
    }
  }
  return getLatestActiveDay();
}

async function bootRoot(): Promise<void> {
  const activeDay = resolveRootTargetDay();
  const targetKey = `./days/${activeDay.slug}/main.ts`;
  const loader = dayModules[targetKey] ?? dayModules['./days/00-specimen/main.ts'];

  if (!loader) {
    throw new Error(`No day module found for ${targetKey}`);
  }

  await loader();
}

void bootRoot();
