import {
  CALIBRATION_DAY,
  DAYS_MANIFEST,
  type DayEntry,
  getAdjacentCompletedDays,
  getDayByNumber,
} from './manifest';
import './shell.css';

export interface MountedDayShell {
  day: DayEntry;
  canvas: HTMLCanvasElement;
  headerButton: HTMLButtonElement;
  drawerBackdrop: HTMLElement;
  openDrawer: () => void;
  closeDrawer: () => void;
  toggleDrawer: () => void;
  registerDispose: (fn: () => void) => void;
  destroy: () => void;
}

export function mountDayShell(dayNumberOrEntry: number | DayEntry): MountedDayShell {
  const day =
    typeof dayNumberOrEntry === 'number' ? getDayByNumber(dayNumberOrEntry) : dayNumberOrEntry;

  document.title = day.headerLabel;

  let appRoot = document.getElementById('app');
  if (!appRoot) {
    appRoot = document.createElement('div');
    appRoot.id = 'app';
    document.body.appendChild(appRoot);
  }
  appRoot.innerHTML = '';

  const canvas = document.createElement('canvas');
  canvas.id = 'specimen-canvas';
  canvas.setAttribute('aria-label', `${day.headerLabel} — ${day.specimenSummary}`);
  appRoot.appendChild(canvas);

  const headerButton = document.createElement('button');
  headerButton.type = 'button';
  headerButton.className = 'day-header';
  headerButton.setAttribute('aria-expanded', 'false');
  headerButton.setAttribute('aria-controls', 'day-index-drawer');
  headerButton.textContent = day.headerLabel;
  appRoot.appendChild(headerButton);

  // Portaled 30-day drawer mounted directly onto document.body (z-index: 1000)
  const existingDrawer = document.getElementById('day-index-drawer');
  if (existingDrawer) {
    existingDrawer.remove();
  }

  const drawerBackdrop = document.createElement('nav');
  drawerBackdrop.id = 'day-index-drawer';
  drawerBackdrop.className = 'day-drawer-backdrop';
  drawerBackdrop.setAttribute('aria-label', 'Codevember 30-Day Index');

  const grid = document.createElement('div');
  grid.className = 'day-drawer-grid';

  const allEntries: DayEntry[] =
    day.day === 0 ? [CALIBRATION_DAY, ...DAYS_MANIFEST] : [...DAYS_MANIFEST];

  for (const entry of allEntries) {
    const isComplete = entry.status === 'complete';
    const isCurrent = entry.day === day.day;

    const item = document.createElement(isComplete ? 'a' : 'span');
    item.className = 'day-drawer-item';
    if (isComplete) {
      item.classList.add('is-complete');
      (item as HTMLAnchorElement).href = `/${entry.slug}/`;
    }
    if (isCurrent) {
      item.classList.add('is-current');
      item.setAttribute('aria-current', 'page');
    }

    const numSpan = document.createElement('span');
    numSpan.className = 'day-drawer-num';
    numSpan.textContent = `Day ${entry.pad}`;

    const titleSpan = document.createElement('span');
    titleSpan.textContent = `— ${entry.title}`;

    item.appendChild(numSpan);
    item.appendChild(titleSpan);
    grid.appendChild(item);
  }

  drawerBackdrop.appendChild(grid);
  document.body.appendChild(drawerBackdrop);

  let isOpen = false;
  const openDrawer = () => {
    isOpen = true;
    headerButton.setAttribute('aria-expanded', 'true');
    drawerBackdrop.classList.add('is-open');
  };

  const closeDrawer = () => {
    isOpen = false;
    headerButton.setAttribute('aria-expanded', 'false');
    drawerBackdrop.classList.remove('is-open');
  };

  const toggleDrawer = () => {
    if (isOpen) {
      closeDrawer();
    } else {
      openDrawer();
    }
  };

  const onHeaderClick = (ev: MouseEvent) => {
    ev.stopPropagation();
    toggleDrawer();
  };

  const onBackdropClick = (ev: MouseEvent) => {
    if (ev.target === drawerBackdrop) {
      closeDrawer();
    }
  };

  const onKeyDown = (ev: KeyboardEvent) => {
    if (ev.key === 'Escape' && isOpen) {
      closeDrawer();
      return;
    }
    if (ev.defaultPrevented || ev.target instanceof HTMLInputElement) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') {
      const { prev, next } = getAdjacentCompletedDays(day.day);
      const target = ev.key === 'ArrowLeft' ? prev : next;
      if (target) {
        window.location.href = `/${target.slug}/`;
      }
    }
  };

  headerButton.addEventListener('click', onHeaderClick);
  drawerBackdrop.addEventListener('click', onBackdropClick);
  window.addEventListener('keydown', onKeyDown);

  const disposeCallbacks: Array<() => void> = [];
  let disposed = false;

  const runDispose = () => {
    if (disposed) return;
    disposed = true;
    for (const fn of disposeCallbacks) {
      try {
        fn();
      } catch {
        // Ignore teardown errors during page unload
      }
    }
  };

  const onPageHide = (ev: PageTransitionEvent) => {
    if (!ev.persisted) {
      runDispose();
    }
  };

  window.addEventListener('pagehide', onPageHide);

  return {
    day,
    canvas,
    headerButton,
    drawerBackdrop,
    openDrawer,
    closeDrawer,
    toggleDrawer,
    registerDispose: (fn: () => void) => {
      disposeCallbacks.push(fn);
    },
    destroy: () => {
      runDispose();
      headerButton.removeEventListener('click', onHeaderClick);
      drawerBackdrop.removeEventListener('click', onBackdropClick);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('pagehide', onPageHide);
      drawerBackdrop.remove();
    },
  };
}
