import * as THREE from 'three/webgpu';

/**
 * Zero-GC Tactile Pointer Field Tracker (`src/core/stage/PointerField.ts`)
 *
 * Tracks normalized device coordinates, 3D raycast intersection against the
 * specimen's focal plane, EMA-smoothed pointer velocity, physical dwell time
 * (`dwellTimeSec`), and click impulse decay without allocating inside `rAF`.
 */
export class PointerField {
  readonly ndc = new THREE.Vector2(0, 0);
  readonly prevNdc = new THREE.Vector2(0, 0);
  readonly worldPos = new THREE.Vector3(0, 0, 0);
  readonly prevWorldPos = new THREE.Vector3(0, 0, 0);
  readonly velocity = new THREE.Vector3(0, 0, 0);

  /** Smoothed scalar pointer speed in world units / sec */
  speed = 0;
  /** Physical seconds pointer has dwelled/held near current spot */
  dwellTimeSec = 0;
  /** Transient click/press impulse in [0, 1] decaying exponentially */
  impulse = 0;
  /** True while primary pointer button is depressed */
  isDown = false;
  /** True once the pointer has entered the canvas at least once */
  isActive = false;

  // Pre-allocated TSL uniforms for direct shader binding (zero per-frame allocation)
  readonly uWorldPos = new THREE.Vector3(0, 0, 0);
  readonly uVelocity = new THREE.Vector3(0, 0, 0);
  readonly uMeta = new THREE.Vector4(0, 0, 0, 0); // x: speed, y: dwellTimeSec, z: impulse, w: isDown (1|0)

  private readonly raycaster = new THREE.Raycaster();
  private readonly interactionPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  private readonly planeNormal = new THREE.Vector3(0, 0, 1);
  private readonly intersectScratch = new THREE.Vector3();
  private readonly deltaScratch = new THREE.Vector3();

  private readonly element: HTMLElement;
  private readonly onMoveBound: (ev: PointerEvent) => void;
  private readonly onDownBound: (ev: PointerEvent) => void;
  private readonly onUpBound: (ev: PointerEvent) => void;
  private readonly onLeaveBound: () => void;

  constructor(element: HTMLElement) {
    this.element = element;
    this.onMoveBound = (ev: PointerEvent) => this.handlePointerMove(ev);
    this.onDownBound = (ev: PointerEvent) => this.handlePointerDown(ev);
    this.onUpBound = () => this.handlePointerUp();
    this.onLeaveBound = () => {
      this.isDown = false;
    };

    element.addEventListener('pointermove', this.onMoveBound, { passive: true });
    element.addEventListener('pointerdown', this.onDownBound, { passive: true });
    window.addEventListener('pointerup', this.onUpBound, { passive: true });
    element.addEventListener('pointerleave', this.onLeaveBound, { passive: true });
  }

  private updateNdcFromEvent(ev: PointerEvent): void {
    const rect = this.element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const events = typeof ev.getCoalescedEvents === 'function' ? ev.getCoalescedEvents() : [ev];
    const latest = events.length > 0 ? events[events.length - 1] : ev;

    const nx = ((latest.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -((latest.clientY - rect.top) / rect.height) * 2 + 1;
    this.ndc.set(Math.max(-1, Math.min(1, nx)), Math.max(-1, Math.min(1, ny)));
    this.isActive = true;
  }

  private handlePointerMove(ev: PointerEvent): void {
    this.updateNdcFromEvent(ev);
  }

  private handlePointerDown(ev: PointerEvent): void {
    this.updateNdcFromEvent(ev);
    this.isDown = true;
    this.impulse = 1.0;
  }

  private handlePointerUp(): void {
    this.isDown = false;
  }

  /**
   * Advances pointer kinematics by physical `dtSec` and projects current NDC
   * onto the camera-facing plane passing through the world origin.
   */
  update(camera: THREE.Camera, dtSec: number): void {
    const dt = Math.max(0.001, Math.min(0.1, dtSec));

    camera.getWorldDirection(this.planeNormal).negate();
    this.interactionPlane.setFromNormalAndCoplanarPoint(this.planeNormal, THREE.Object3D.DEFAULT_UP.set(0, 0, 0));
    THREE.Object3D.DEFAULT_UP.set(0, 1, 0);

    this.raycaster.setFromCamera(this.ndc, camera);
    const hit = this.raycaster.ray.intersectPlane(this.interactionPlane, this.intersectScratch);
    if (hit) {
      if (!this.isActive) {
        this.prevWorldPos.copy(this.intersectScratch);
      }
      this.worldPos.copy(this.intersectScratch);
    }

    this.deltaScratch.subVectors(this.worldPos, this.prevWorldPos).divideScalar(dt);
    const instSpeed = this.deltaScratch.length();

    // EMA low-pass smoothing on velocity and scalar speed
    const alpha = 1 - Math.exp(-dt * 18);
    this.velocity.lerp(this.deltaScratch, alpha);
    this.speed = this.speed + (instSpeed - this.speed) * alpha;

    // Velocity-gated dwell accumulation (drains cleanly when moving fast)
    if (this.isActive && (this.isDown || this.speed < 0.35)) {
      const rate = this.isDown ? 1.6 : 0.45;
      this.dwellTimeSec = Math.min(4.0, this.dwellTimeSec + dt * rate);
    } else {
      this.dwellTimeSec = Math.max(0, this.dwellTimeSec - dt * 1.8);
    }

    // Exponential decay on click impulse
    this.impulse *= Math.exp(-dt * 4.5);
    if (this.impulse < 1e-4) this.impulse = 0;

    this.prevWorldPos.copy(this.worldPos);
    this.prevNdc.copy(this.ndc);

    // Sync pre-allocated shader uniform vectors
    this.uWorldPos.copy(this.worldPos);
    this.uVelocity.copy(this.velocity);
    this.uMeta.set(this.speed, this.dwellTimeSec, this.impulse, this.isDown ? 1.0 : 0.0);
  }

  dispose(): void {
    this.element.removeEventListener('pointermove', this.onMoveBound);
    this.element.removeEventListener('pointerdown', this.onDownBound);
    window.removeEventListener('pointerup', this.onUpBound);
    this.element.removeEventListener('pointerleave', this.onLeaveBound);
  }
}
