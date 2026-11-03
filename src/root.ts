import { getLatestActiveDay } from './core/manifest';

type DayModuleLoader = () => Promise<Record<string, unknown>>;

const dayModules = import.meta.glob('./days/*/main.ts') as Record<string, DayModuleLoader>;

async function bootRoot(): Promise<void> {
  const activeDay = getLatestActiveDay();
  const targetKey = `./days/${activeDay.slug}/main.ts`;
  const loader = dayModules[targetKey] ?? dayModules['./days/00-specimen/main.ts'];

  if (!loader) {
    throw new Error(`No day module found for ${targetKey}`);
  }

  await loader();
}

void bootRoot();
