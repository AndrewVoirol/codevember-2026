import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three/webgpu';
import { getDayByNumber } from '../../core/manifest';
import { SpecimenStage } from '../../core/stage/SpecimenStage';
import {
  buildDowelGeometry,
  buildDustMoteGeometry,
  buildPebbleEggGeometry,
  buildPlinthBrassInnerBezelGeometry,
  buildPlinthBrassOuterLipGeometry,
  buildPlinthLimestoneGeometry,
} from './nestGeometry';
import { createNestMaterials, createNestStudioEnvironment } from './nestMaterials';
import {
  MAX_DUST_MOTES,
  NestDustPool,
  NestReweaveController,
  ReciprocalNestSim,
  type Day24TelemetryState,
} from './nestSimulation';
import {
  computeAllRodContactStations,
  generateReciprocalDomeLayout,
} from './reciprocalDome';

declare global {
  interface Window {
    __DAY24__?: {
      getState: () => Day24TelemetryState;
    };
  }
}

interface TierInstancedGroup {
  readonly tier: number;
  readonly rodIds: readonly number[];
  readonly count: number;
  readonly mesh: THREE.InstancedMesh;
  readonly geometry: THREE.BufferGeometry;
  readonly metaArray: Float32Array;
  readonly metaAttr: THREE.InstancedBufferAttribute;
  readonly quatArray: Float32Array;
  readonly quatAttr: THREE.InstancedBufferAttribute;
}

export async function bootDay24Nest(): Promise<SpecimenStage> {
  await RAPIER.init();

  const dayEntry = getDayByNumber(24);
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 1.92, 3.84],
    fov: 35,
    enableOrbit: true,
    enableShadows: true,
  });

  // Hero museum 3/4 framing centered on the suspended ring + woven timber dome + ovoid celadon eggs
  stage.controls.target.set(0, 0.15, 0);
  stage.controls.minDistance = 0.85;
  stage.controls.maxDistance = 7.8;
  stage.controls.minPolarAngle = 0.08;
  stage.controls.maxPolarAngle = Math.PI * 0.94;
  stage.controls.update();

  // Studio softbox PMREM environment map for dark-brass, oiled walnut & satin celadon reflections
  const studioEnv = createNestStudioEnvironment(stage.renderer);
  stage.scene.environment = studioEnv.texture;
  stage.scene.environmentIntensity = 0.98;

  // ---------------------------------------------------------------------------
  // Studio Shadow Map & Sculptural Lattice Lighting Calibration
  // Upgrade SpecimenStage's default 1024 shadow map (d=2.4) to crisp 2048x2048
  // tight-frustum (d=1.65 -> 1.6 mm/texel) interlaced shadows with zero acne/peter-panning.
  // ---------------------------------------------------------------------------
  const dirLights: THREE.DirectionalLight[] = [];
  for (let i = 0; i < stage.scene.children.length; i++) {
    const child = stage.scene.children[i];
    if (child instanceof THREE.DirectionalLight) {
      dirLights.push(child);
    } else if (child instanceof THREE.HemisphereLight) {
      child.intensity = 0.64;
    }
  }

  const keyLight = dirLights.find((l) => l.castShadow);
  if (keyLight) {
    // Raking warm studio key light draping crisp interlaced lattice shadows across timber, limestone & eggs
    keyLight.color.setHex(0xf8f1e4);
    keyLight.intensity = 2.62;
    keyLight.position.set(3.05, 4.05, 2.55);
    keyLight.target.position.set(0, 0.22, 0);
    stage.scene.add(keyLight.target);

    if (keyLight.shadow.map) {
      keyLight.shadow.map.dispose();
      keyLight.shadow.map = null;
    }
    keyLight.shadow.mapSize.set(2048, 2048);
    const d = 1.65;
    keyLight.shadow.camera.left = -d;
    keyLight.shadow.camera.right = d;
    keyLight.shadow.camera.top = d;
    keyLight.shadow.camera.bottom = -d;
    keyLight.shadow.camera.near = 1.5;
    keyLight.shadow.camera.far = 10.5;
    keyLight.shadow.bias = -0.00018;
    keyLight.shadow.normalBias = 0.008;
    keyLight.shadow.radius = 2.0;
    keyLight.shadow.camera.updateProjectionMatrix();
  }

  for (let i = 0; i < dirLights.length; i++) {
    const light = dirLights[i];
    if (light === keyLight) continue;
    if (light.position.z < 0) {
      // Grazing back-top calcite rim light separating crown twigs & eggs cleanly from #212529
      light.color.setHex(0xf4efe5);
      light.intensity = 1.68;
      light.position.set(-1.65, 3.45, -3.55);
    } else {
      // Subtle cool mineral-slate fill light preserving shadow-side grain depth without flattening
      light.color.setHex(0x8fa4b4);
      light.intensity = 0.72;
      light.position.set(-3.4, 0.4, 2.2);
    }
  }

  // Soft warm studio uplift beneath the hollow plinth ring so underside weaving reads crisply
  // without stamping near-field specular pinhead dots onto the satin river-pebble eggs
  const undersideBounce = new THREE.PointLight(0xeae3d2, 0.92, 8.5, 1.45);
  undersideBounce.position.set(0.35, -2.65, 0.45);
  stage.specimenGroup.add(undersideBounce);

  // Broad warm bounce light reflected upward off the honed travertine shelf into the underside
  // of the hollow timber dome lattice (zero harsh point-light specular hotspots on the eggs)
  const limestoneBounce = new THREE.DirectionalLight(0xf1e4ca, 0.46);
  limestoneBounce.position.set(0.45, -2.4, 0.35);
  stage.specimenGroup.add(limestoneBounce);

  // Build exact analytical zero-overlap 56-rod reciprocal dome + 3 ovoid Celadon river-pebble eggs
  const layout = generateReciprocalDomeLayout();
  const sim = new ReciprocalNestSim(RAPIER, layout);
  const reweave = new NestReweaveController(layout);
  const dustPool = new NestDustPool(MAX_DUST_MOTES);
  const mats = createNestMaterials(layout.eggs);

  const nestGroup = new THREE.Group();
  nestGroup.name = 'ReciprocalTimberNestSpecimen';
  stage.specimenGroup.add(nestGroup);

  // ---------------------------------------------------------------------------
  // 1. Suspended Honed-Limestone & Dark-Brass Circular Plinth Ring (Hollow Center)
  // ---------------------------------------------------------------------------
  const plinthLimestoneGeom = buildPlinthLimestoneGeometry(layout.plinth);
  const plinthLimestoneMesh = new THREE.Mesh(plinthLimestoneGeom, mats.limestoneMat);
  plinthLimestoneMesh.name = 'PlinthLimestoneRing';
  plinthLimestoneMesh.castShadow = true;
  plinthLimestoneMesh.receiveShadow = true;
  nestGroup.add(plinthLimestoneMesh);

  const plinthBrassLipGeom = buildPlinthBrassOuterLipGeometry(layout.plinth);
  const plinthBrassLipMesh = new THREE.Mesh(plinthBrassLipGeom, mats.brassMat);
  plinthBrassLipMesh.name = 'PlinthDarkBrassOuterLip';
  plinthBrassLipMesh.castShadow = true;
  plinthBrassLipMesh.receiveShadow = true;
  nestGroup.add(plinthBrassLipMesh);

  const plinthBrassInnerGeom = buildPlinthBrassInnerBezelGeometry(layout.plinth);
  const plinthBrassInnerMesh = new THREE.Mesh(plinthBrassInnerGeom, mats.brassMat);
  plinthBrassInnerMesh.name = 'PlinthDarkBrassInnerBezel';
  plinthBrassInnerMesh.castShadow = true;
  plinthBrassInnerMesh.receiveShadow = true;
  nestGroup.add(plinthBrassInnerMesh);

  // ---------------------------------------------------------------------------
  // 2. Per-Tier / Per-Symmetry InstancedMesh Groups (56 Chamfered Flat-Ended Dowels)
  //    Every bucket bakes ALL real same-tier + inter-tier crossing burnish stations
  //    (splits Tier-1's 16 rods cleanly into 2 x 8-rod even/odd buckets for 0-mismatch patina)
  // ---------------------------------------------------------------------------
  const tierGroups: TierInstancedGroup[] = [];
  const tierMeshes: THREE.InstancedMesh[] = [];
  const meshToTierGroup = new Map<THREE.Object3D, TierInstancedGroup>();

  let rodCursor = 0;
  for (let t = 0; t < layout.tiers.length; t++) {
    const tierMeta = layout.tiers[t];
    const bucketList: Array<{ rodsInBucket: typeof layout.rods[number][]; stations: number[] }> = [];
    const bucketBySig = new Map<string, number>();

    for (let i = 0; i < tierMeta.count; i++) {
      const rod = layout.rods[rodCursor + i];
      const stations = computeAllRodContactStations(rod, layout.rods);
      const sig = stations.map((s) => s.toFixed(3)).join(',');
      let bIdx = bucketBySig.get(sig);
      if (bIdx === undefined) {
        bIdx = bucketList.length;
        bucketBySig.set(sig, bIdx);
        bucketList.push({ rodsInBucket: [], stations });
      }
      bucketList[bIdx].rodsInBucket.push(rod);
    }

    for (let bIdx = 0; bIdx < bucketList.length; bIdx++) {
      const bucket = bucketList[bIdx];
      const rodsInBucket = bucket.rodsInBucket;
      const firstRod = rodsInBucket[0];
      const geom = buildDowelGeometry(
        tierMeta.halfHeight,
        tierMeta.radius,
        firstRod.notchParams,
        28,
        bucket.stations
      );

      const count = rodsInBucket.length;
      const metaArray = new Float32Array(count * 4);
      const quatArray = new Float32Array(count * 4);
      const rodIds: number[] = [];
      for (let i = 0; i < count; i++) {
        const r = rodsInBucket[i];
        rodIds.push(r.id);
        metaArray[i * 4 + 0] = r.woodSpecies;
        metaArray[i * 4 + 1] = r.grainSeed;
        metaArray[i * 4 + 2] = r.id;
        metaArray[i * 4 + 3] = 0.0;

        quatArray[i * 4 + 0] = r.quat.x;
        quatArray[i * 4 + 1] = r.quat.y;
        quatArray[i * 4 + 2] = r.quat.z;
        quatArray[i * 4 + 3] = r.quat.w;
      }
      const metaAttr = new THREE.InstancedBufferAttribute(metaArray, 4);
      metaAttr.setUsage(THREE.DynamicDrawUsage);
      geom.setAttribute('aRodMeta', metaAttr);

      const quatAttr = new THREE.InstancedBufferAttribute(quatArray, 4);
      quatAttr.setUsage(THREE.DynamicDrawUsage);
      geom.setAttribute('aRodQuat', quatAttr);

      const instMesh = new THREE.InstancedMesh(geom, mats.timberMat, count);
      instMesh.name =
        bucketList.length === 1
          ? `ReciprocalTimberTier${t}`
          : `ReciprocalTimberTier${t}_${bIdx}`;
      instMesh.castShadow = true;
      instMesh.receiveShadow = true;
      instMesh.frustumCulled = false;
      nestGroup.add(instMesh);

      const group: TierInstancedGroup = {
        tier: t,
        rodIds,
        count,
        mesh: instMesh,
        geometry: geom,
        metaArray,
        metaAttr,
        quatArray,
        quatAttr,
      };
      tierGroups.push(group);
      tierMeshes.push(instMesh);
      meshToTierGroup.set(instMesh, group);
    }
    rodCursor += tierMeta.count;
  }

  // ---------------------------------------------------------------------------
  // 3. Three Speckled Celadon & Bone-Calcite River-Pebble Ovoid "Eggs"
  // ---------------------------------------------------------------------------
  const eggGeoms: THREE.BufferGeometry[] = [];
  const eggMeshes: THREE.Mesh[] = [];
  for (let i = 0; i < layout.eggs.length; i++) {
    const eggSpec = layout.eggs[i];
    const geom = buildPebbleEggGeometry(eggSpec.radius, eggSpec.visualScale, eggSpec.seed, 4);
    const mesh = new THREE.Mesh(geom, mats.eggMats[i]);
    mesh.name = `CeladonPebbleEgg${i}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    nestGroup.add(mesh);
    eggGeoms.push(geom);
    eggMeshes.push(mesh);
  }

  // ---------------------------------------------------------------------------
  // 4. Pre-Allocated Instanced Dry Wood-Dust & Limestone-Chalk Mote Pool
  // ---------------------------------------------------------------------------
  const dustGeom = buildDustMoteGeometry();
  const dustMetaArray = new Float32Array(MAX_DUST_MOTES * 4);
  const dustMetaAttr = new THREE.InstancedBufferAttribute(dustMetaArray, 4);
  dustMetaAttr.setUsage(THREE.DynamicDrawUsage);
  dustGeom.setAttribute('aDustMeta', dustMetaAttr);

  const dustMesh = new THREE.InstancedMesh(dustGeom, mats.dustMat, MAX_DUST_MOTES);
  dustMesh.name = 'NestDryDustMotes';
  dustMesh.castShadow = false;
  dustMesh.receiveShadow = false;
  dustMesh.frustumCulled = false;
  nestGroup.add(dustMesh);

  // ---------------------------------------------------------------------------
  // 5. Pre-Allocated Scratch Objects (Strictly Zero Per-Frame Allocations)
  // ---------------------------------------------------------------------------
  const scratchPos = new THREE.Vector3();
  const scratchQuat = new THREE.Quaternion();
  const scratchQuatInv = new THREE.Quaternion();
  const scratchEuler = new THREE.Euler();
  const scratchScale = new THREE.Vector3(1, 1, 1);
  const scratchZeroScale = new THREE.Vector3(0, 0, 0);
  const scratchDustScale = new THREE.Vector3(1, 1, 1);
  const scratchMat4 = new THREE.Matrix4();

  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  const camForward = new THREE.Vector3();
  const camRight = new THREE.Vector3();
  const camUp = new THREE.Vector3();
  const dragPlane = new THREE.Plane();
  const dragTargetVec = new THREE.Vector3();
  const localHitVec = new THREE.Vector3();
  const pokeDirVec = new THREE.Vector3();
  const sweepDirVec = new THREE.Vector3();
  const hitWorldVec = new THREE.Vector3();

  const rodFlash = new Float32Array(layout.rods.length);
  const prevRodVelY = new Float32Array(layout.rods.length);
  const eggPosUniforms = [mats.uEggPos0, mats.uEggPos1, mats.uEggPos2] as const;

  let dustHadActive = true;

  const syncEggContactUniforms = () => {
    for (let i = 0; i < eggMeshes.length; i++) {
      const m = eggMeshes[i];
      eggPosUniforms[i].value.copy(m.position);
    }
    const c0 = Math.max(0, Math.min(1, (eggMeshes[0].position.y - 0.45) / 0.28));
    const c1 = Math.max(0, Math.min(1, (eggMeshes[1].position.y - 0.45) / 0.28));
    const c2 = Math.max(0, Math.min(1, (eggMeshes[2].position.y - 0.45) / 0.28));
    mats.uEggCradleAo.value.set(c0, c1, c2);
  };

  // Sync all 56 timber instances + 3 ovoid egg meshes directly from Rapier world transforms
  const syncMeshesFromPhysics = () => {
    for (let t = 0; t < tierGroups.length; t++) {
      const grp = tierGroups[t];
      let metaDirty = false;
      for (let i = 0; i < grp.count; i++) {
        const rodId = grp.rodIds[i];
        const rec = sim.rods[rodId];
        if (rec.removed || rec.abyssCulled) {
          scratchPos.set(0, -100, 0);
          scratchQuat.identity();
          scratchMat4.compose(scratchPos, scratchQuat, scratchZeroScale);
          grp.quatArray[i * 4 + 0] = 0;
          grp.quatArray[i * 4 + 1] = 0;
          grp.quatArray[i * 4 + 2] = 0;
          grp.quatArray[i * 4 + 3] = 1;
        } else {
          const tr = rec.body.translation();
          const rq = rec.body.rotation();
          scratchPos.set(tr.x, tr.y, tr.z);
          scratchQuat.set(rq.x, rq.y, rq.z, rq.w);
          scratchMat4.compose(scratchPos, scratchQuat, scratchScale);
          grp.quatArray[i * 4 + 0] = rq.x;
          grp.quatArray[i * 4 + 1] = rq.y;
          grp.quatArray[i * 4 + 2] = rq.z;
          grp.quatArray[i * 4 + 3] = rq.w;
        }
        grp.mesh.setMatrixAt(i, scratchMat4);

        const prevFlash = grp.metaArray[i * 4 + 3];
        const nextFlash = rodFlash[rodId];
        if (Math.abs(prevFlash - nextFlash) > 1e-4) {
          grp.metaArray[i * 4 + 3] = nextFlash;
          metaDirty = true;
        }
      }
      grp.mesh.instanceMatrix.needsUpdate = true;
      grp.quatAttr.needsUpdate = true;
      if (metaDirty) {
        grp.metaAttr.needsUpdate = true;
      }
    }

    for (let i = 0; i < sim.eggs.length; i++) {
      const eggRec = sim.eggs[i];
      const mesh = eggMeshes[i];
      mesh.visible = !eggRec.abyssCulled;
      if (eggRec.abyssCulled) continue;
      const tr = eggRec.body.translation();
      const rq = eggRec.body.rotation();
      mesh.position.set(tr.x, tr.y, tr.z);
      mesh.quaternion.set(rq.x, rq.y, rq.z, rq.w);
    }

    syncEggContactUniforms();
  };

  // Sync all 56 timber instances + 3 ovoid eggs smoothly during choreographed re-weave
  const syncMeshesFromReweave = () => {
    for (let t = 0; t < tierGroups.length; t++) {
      const grp = tierGroups[t];
      let metaDirty = false;
      for (let i = 0; i < grp.count; i++) {
        const rodId = grp.rodIds[i];
        const pOff = rodId * 3;
        const qOff = rodId * 4;
        scratchPos.set(
          reweave.rodCurrentPos[pOff + 0],
          reweave.rodCurrentPos[pOff + 1],
          reweave.rodCurrentPos[pOff + 2]
        );
        const qx = reweave.rodCurrentQuat[qOff + 0];
        const qy = reweave.rodCurrentQuat[qOff + 1];
        const qz = reweave.rodCurrentQuat[qOff + 2];
        const qw = reweave.rodCurrentQuat[qOff + 3];
        scratchQuat.set(qx, qy, qz, qw);
        scratchMat4.compose(scratchPos, scratchQuat, scratchScale);
        grp.mesh.setMatrixAt(i, scratchMat4);

        grp.quatArray[i * 4 + 0] = qx;
        grp.quatArray[i * 4 + 1] = qy;
        grp.quatArray[i * 4 + 2] = qz;
        grp.quatArray[i * 4 + 3] = qw;

        if (grp.metaArray[i * 4 + 3] !== 0) {
          grp.metaArray[i * 4 + 3] = 0;
          metaDirty = true;
        }
      }
      grp.mesh.instanceMatrix.needsUpdate = true;
      grp.quatAttr.needsUpdate = true;
      if (metaDirty) grp.metaAttr.needsUpdate = true;
    }

    for (let i = 0; i < eggMeshes.length; i++) {
      const pOff = i * 3;
      const qOff = i * 4;
      const mesh = eggMeshes[i];
      mesh.visible = true;
      mesh.position.set(
        reweave.eggCurrentPos[pOff + 0],
        reweave.eggCurrentPos[pOff + 1],
        reweave.eggCurrentPos[pOff + 2]
      );
      mesh.quaternion.set(
        reweave.eggCurrentQuat[qOff + 0],
        reweave.eggCurrentQuat[qOff + 1],
        reweave.eggCurrentQuat[qOff + 2],
        reweave.eggCurrentQuat[qOff + 3]
      );
    }

    syncEggContactUniforms();
  };

  const syncDustMesh = () => {
    const activeCount = dustPool.getActiveCount();
    if (activeCount === 0 && !dustHadActive) return;

    for (let i = 0; i < dustPool.capacity; i++) {
      if (!dustPool.active[i]) {
        scratchPos.set(0, -100, 0);
        scratchQuat.identity();
        scratchMat4.compose(scratchPos, scratchQuat, scratchZeroScale);
        dustMesh.setMatrixAt(i, scratchMat4);
        dustMetaArray[i * 4 + 1] = 0;
      } else {
        const pOff = i * 3;
        scratchPos.set(
          dustPool.pos[pOff + 0],
          dustPool.pos[pOff + 1],
          dustPool.pos[pOff + 2]
        );
        scratchEuler.set(
          dustPool.rot[pOff + 0],
          dustPool.rot[pOff + 1],
          dustPool.rot[pOff + 2]
        );
        scratchQuat.setFromEuler(scratchEuler);
        const s = dustPool.size[i];
        scratchDustScale.set(s, s, s);
        scratchMat4.compose(scratchPos, scratchQuat, scratchDustScale);
        dustMesh.setMatrixAt(i, scratchMat4);

        dustMetaArray[i * 4 + 0] = dustPool.tone[i];
        dustMetaArray[i * 4 + 1] = dustPool.opacity[i];
        dustMetaArray[i * 4 + 2] = i * 0.015625;
        dustMetaArray[i * 4 + 3] = s;
      }
    }
    dustMesh.instanceMatrix.needsUpdate = true;
    dustMetaAttr.needsUpdate = true;
    dustHadActive = activeCount > 0;
  };

  syncMeshesFromPhysics();
  syncDustMesh();

  // ---------------------------------------------------------------------------
  // 6. Tactile Jenga Interaction: Warm Subtle Hover Rim, Silky Axial-Biased Grab,
  //    Load-Aware Tap-Test + Dust Motes, 360° Orbit & Choreographed Spiral Re-Weave
  // ---------------------------------------------------------------------------
  const canvas = stage.shell.canvas;
  let hoveredRodId = -1;
  let hoverLiftCurrent = 0;
  let grabbingRodId = -1;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastClientX = 0;
  let lastClientY = 0;
  let dragTravelPx = 0;
  let lastSweepNudgeMs = 0;
  let lastScrapeDustMs = 0;
  let lastImpactDustMs = 0;

  // Boot-settle cursor guard (ignores stationary parked OS cursor on initial load)
  const bootTimeMs = performance.now();
  let hoverUnlocked = false;
  let firstHoverX: number | null = null;
  let firstHoverY: number | null = null;

  const updatePointerNdc = (clientX: number, clientY: number) => {
    const rect = canvas.getBoundingClientRect();
    pointerNdc.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    pointerNdc.y = -((clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  };

  const pickOffsets: ReadonlyArray<readonly [number, number]> = [
    [0, 0],
    [4, 0],
    [-4, 0],
    [0, 4],
    [0, -4],
  ];

  const pickRodHit = (
    clientX: number,
    clientY: number,
    allowMicroHalo = false
  ): { rodId: number; point: THREE.Vector3 } | null => {
    // Refresh instanced bounding spheres right before picking so displaced/pulled rods remain 100% pickable
    for (let t = 0; t < tierMeshes.length; t++) {
      tierMeshes[t].computeBoundingSphere();
    }
    const probeCount = allowMicroHalo ? pickOffsets.length : 1;
    for (let p = 0; p < probeCount; p++) {
      const [ox, oy] = pickOffsets[p];
      updatePointerNdc(clientX + ox, clientY + oy);
      raycaster.setFromCamera(pointerNdc, stage.camera);
      const hits = raycaster.intersectObjects(tierMeshes, false);
      for (let i = 0; i < hits.length; i++) {
        const h = hits[i];
        if (h.instanceId === undefined) continue;
        const grp = meshToTierGroup.get(h.object);
        if (!grp) continue;
        const rodId = grp.rodIds[h.instanceId];
        if (rodId === undefined) continue;
        const rec = sim.rods[rodId];
        if (!rec || rec.removed || rec.abyssCulled) continue;
        hitWorldVec.copy(h.point);
        return { rodId, point: hitWorldVec };
      }
    }
    return null;
  };

  const startChoreographedReset = () => {
    if (reweave.isActive()) return;
    grabbingRodId = -1;
    hoveredRodId = -1;
    hoverLiftCurrent = 0;
    rodFlash.fill(0);
    sim.endGrab();
    mats.uHoveredRodId.value = -1;
    mats.uGrabbedRodId.value = -1;
    mats.uHoverLift.value = 0;
    stage.controls.enabled = true;
    canvas.style.cursor = 'default';
    reweave.beginFromSim(sim);
    syncMeshesFromReweave();
  };

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0 || reweave.isActive()) return;
    hoverUnlocked = true;
    const hit = pickRodHit(e.clientX, e.clientY, true);
    if (!hit) return;

    const rec = sim.rods[hit.rodId];
    if (!rec || rec.removed) return;

    grabbingRodId = hit.rodId;
    hoveredRodId = hit.rodId;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    dragTravelPx = 0;

    stage.controls.enabled = false;
    canvas.style.cursor = 'grabbing';
    mats.uGrabbedRodId.value = hit.rodId;
    mats.uHoveredRodId.value = hit.rodId;

    // Transform hit point into the rod rigid body's local frame
    const tr = rec.body.translation();
    const rq = rec.body.rotation();
    scratchQuatInv.set(rq.x, rq.y, rq.z, rq.w).invert();
    localHitVec.set(hit.point.x - tr.x, hit.point.y - tr.y, hit.point.z - tr.z);
    localHitVec.applyQuaternion(scratchQuatInv);

    // Anchor camera-facing 3D drag plane through the exact picked surface point
    stage.camera.getWorldDirection(camForward);
    dragPlane.setFromNormalAndCoplanarPoint(camForward, hit.point);
    dragTargetVec.copy(hit.point);

    sim.startGrab(
      hit.rodId,
      [localHitVec.x, localHitVec.y, localHitVec.z],
      [dragTargetVec.x, dragTargetVec.y, dragTargetVec.z]
    );
  };

  const onPointerMove = (e: PointerEvent) => {
    if (reweave.isActive()) return;

    if (grabbingRodId !== -1) {
      const dx = e.clientX - lastClientX;
      const dy = e.clientY - lastClientY;
      dragTravelPx += Math.hypot(dx, dy);
      lastClientX = e.clientX;
      lastClientY = e.clientY;

      updatePointerNdc(e.clientX, e.clientY);
      raycaster.setFromCamera(pointerNdc, stage.camera);
      if (raycaster.ray.intersectPlane(dragPlane, dragTargetVec)) {
        sim.updateGrabTarget([dragTargetVec.x, dragTargetVec.y, dragTargetVec.z]);
      }
      return;
    }

    // Boot-settle guard against stationary OS cursors on initial page load
    if (!hoverUnlocked) {
      if (performance.now() - bootTimeMs < 350) return;
      if (firstHoverX === null || firstHoverY === null) {
        firstHoverX = e.clientX;
        firstHoverY = e.clientY;
        return;
      }
      if (Math.hypot(e.clientX - firstHoverX, e.clientY - firstHoverY) < 8) return;
      hoverUnlocked = true;
    }

    const hit = pickRodHit(e.clientX, e.clientY);
    if (hit) {
      hoveredRodId = hit.rodId;
      mats.uHoveredRodId.value = hit.rodId;
      canvas.style.cursor = 'grab';

      // Faint tactile micro-shift brush on brisk pointer sweeps across timber dowels
      const nowMs = performance.now();
      if (stage.pointer.speed > 0.45 && nowMs - lastSweepNudgeMs > 90) {
        lastSweepNudgeMs = nowMs;
        camRight.set(1, 0, 0).applyQuaternion(stage.camera.quaternion);
        camUp.set(0, 1, 0).applyQuaternion(stage.camera.quaternion);
        sweepDirVec
          .copy(camRight)
          .multiplyScalar(stage.pointer.velocity.x)
          .addScaledVector(camUp, stage.pointer.velocity.y);
        if (sweepDirVec.lengthSq() > 1e-5) {
          sweepDirVec.normalize();
          sim.pokeRod(
            hit.rodId,
            [hit.point.x, hit.point.y, hit.point.z],
            [sweepDirVec.x, sweepDirVec.y, sweepDirVec.z],
            0.00065
          );
          rodFlash[hit.rodId] = Math.max(rodFlash[hit.rodId], 0.16);
        }
      }
    } else {
      if (hoveredRodId !== -1) {
        hoveredRodId = -1;
        canvas.style.cursor = 'default';
      }
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    if (grabbingRodId === -1) return;
    const releasedRodId = grabbingRodId;
    grabbingRodId = -1;
    sim.endGrab();
    mats.uGrabbedRodId.value = -1;
    stage.controls.enabled = true;
    canvas.style.cursor = hoveredRodId !== -1 ? 'grab' : 'default';

    // Quick click on a dowel: load-aware tactile tap-test!
    // Loose dowels give a crisp micro-wiggle & warm dry wood-dust puff;
    // heavily pinned load-bearing dowels resist with a tight muted shudder.
    const totalDist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
    if (dragTravelPx < 8 && totalDist < 8) {
      const rec = sim.rods[releasedRodId];
      const pinFactor = sim.estimateRodPinFactor(releasedRodId);
      const isLoose = pinFactor < 0.45;
      const tapStrength = isLoose ? 0.0031 : 0.0021;

      stage.camera.getWorldDirection(camForward);
      camRight.set(1, 0, 0).applyQuaternion(stage.camera.quaternion);
      pokeDirVec.copy(camForward).multiplyScalar(0.82).addScaledVector(camRight, 0.38).normalize();
      sim.pokeRod(
        releasedRodId,
        [dragTargetVec.x, dragTargetVec.y, dragTargetVec.z],
        [pokeDirVec.x, pokeDirVec.y, pokeDirVec.z],
        tapStrength
      );
      rodFlash[releasedRodId] = isLoose ? 0.64 : 0.34;

      // Subtle dry wood-dust puff scaled by looseness (species-matched pigment tone)
      const dustTone = rec ? rec.spec.woodSpecies * 0.55 : 0.3;
      dustPool.emitBurst(
        dragTargetVec.x,
        dragTargetVec.y + 0.015,
        dragTargetVec.z,
        isLoose ? 7 : 2,
        isLoose ? 0.24 : 0.09,
        dustTone,
        0.022
      );
    }
  };

  const onDoubleClick = (e: MouseEvent) => {
    // Double-clicking empty space triggers the choreographed bottom-to-top spiral re-weave
    const hit = pickRodHit(e.clientX, e.clientY);
    if (!hit) {
      startChoreographedReset();
    }
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.code === 'Space' && !e.repeat) {
      e.preventDefault();
      startChoreographedReset();
    }
  };

  const onWheel = () => {
    hoverUnlocked = true;
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('dblclick', onDoubleClick);
  canvas.addEventListener('wheel', onWheel, { passive: true });
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
  window.addEventListener('keydown', onKeyDown);

  stage.shell.registerDispose(() => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('dblclick', onDoubleClick);
    canvas.removeEventListener('wheel', onWheel);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('pointercancel', onPointerUp);
    window.removeEventListener('keydown', onKeyDown);

    stage.scene.environment = null;
    studioEnv.dispose();
    plinthLimestoneGeom.dispose();
    plinthBrassLipGeom.dispose();
    plinthBrassInnerGeom.dispose();
    dustGeom.dispose();
    for (const grp of tierGroups) grp.geometry.dispose();
    for (const g of eggGeoms) g.dispose();
    mats.dispose();
    sim.dispose();
  });

  // Expose canonical read-only Day 24 telemetry hook
  window.__DAY24__ = {
    getState: () => ({
      ...sim.getState(),
      reweaving: reweave.isActive(),
      reweaveProgress: Number(reweave.getProgress().toFixed(3)),
      dustActiveCount: dustPool.getActiveCount(),
    }),
  };

  // Main render, choreographed re-weave, dust motes & Rapier 0.21 physics loop (zero per-frame allocs)
  stage.onTick(({ dtSec }) => {
    const nowMs = performance.now();

    if (reweave.isActive()) {
      const completed = reweave.step(dtSec);
      dustPool.step(dtSec);
      if (completed) {
        sim.reset();
        prevRodVelY.fill(0);
        syncMeshesFromPhysics();
      } else {
        syncMeshesFromReweave();
      }
      syncDustMesh();
      return;
    }

    sim.step(dtSec);

    // Emit subtle wood-dust motes when a grabbed dowel scrapes free past crossing neighbors
    if (grabbingRodId !== -1 && nowMs - lastScrapeDustMs > 110) {
      const rec = sim.rods[grabbingRodId];
      if (rec && !rec.removed) {
        const lv = rec.body.linvel();
        const tr = rec.body.translation();
        const speed = Math.hypot(lv.x, lv.y, lv.z);
        const rad = Math.hypot(tr.x, tr.z);
        if (speed > 0.16 && rad < 1.42) {
          lastScrapeDustMs = nowMs;
          dustPool.emitBurst(
            tr.x,
            tr.y,
            tr.z,
            3,
            0.18 + Math.min(0.22, speed * 0.25),
            rec.spec.woodSpecies * 0.55,
            0.032
          );
        }
      }
    }

    // Detect tumbling dowel impacts on the honed limestone ring (r ∈ [1.10, 1.38]) -> fine chalk/wood puff
    for (let i = 0; i < sim.rods.length; i++) {
      const rec = sim.rods[i];
      if (rec.removed || rec.abyssCulled) {
        prevRodVelY[i] = 0;
        continue;
      }
      const lv = rec.body.linvel();
      const curVy = lv.y;
      const dvY = curVy - prevRodVelY[i];
      prevRodVelY[i] = curVy;

      if (i !== grabbingRodId && dvY > 0.55 && nowMs - lastImpactDustMs > 85) {
        const tr = rec.body.translation();
        const rad = Math.hypot(tr.x, tr.z);
        if (tr.y > -0.04 && tr.y < 0.16 && rad >= 1.10 && rad <= 1.38) {
          lastImpactDustMs = nowMs;
          dustPool.emitBurst(tr.x, 0.02, tr.z, 6, 0.28, 0.88, 0.035);
        }
      }
    }

    dustPool.step(dtSec);

    // Smooth hover rim fade & tactile poke flash decay (strict zero tint when unhovered)
    const targetHover = hoveredRodId !== -1 ? 1.0 : 0.0;
    const hoverLerp = Math.min(1, dtSec * 14.0);
    hoverLiftCurrent += (targetHover - hoverLiftCurrent) * hoverLerp;
    if (hoveredRodId === -1 && hoverLiftCurrent < 0.002) {
      hoverLiftCurrent = 0;
      mats.uHoveredRodId.value = -1;
    }
    mats.uHoverLift.value = hoverLiftCurrent;

    const flashDecay = Math.exp(-dtSec * 6.5);
    for (let i = 0; i < rodFlash.length; i++) {
      if (rodFlash[i] > 0.002) {
        rodFlash[i] *= flashDecay;
      } else if (rodFlash[i] !== 0) {
        rodFlash[i] = 0;
      }
    }

    syncMeshesFromPhysics();
    syncDustMesh();
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay24Nest();
}
