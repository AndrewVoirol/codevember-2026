import * as THREE from 'three/webgpu';
import {
  clamp,
  cos,
  dot,
  exp,
  float,
  length,
  max,
  mix,
  normalLocal,
  positionLocal,
  sin,
  smoothstep,
  vec3,
} from 'three/tsl';
import { CALIBRATION_DAY } from '../../core/manifest';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

export async function bootDay00Specimen(): Promise<SpecimenStage> {
  const stage = await SpecimenStage.create({
    day: CALIBRATION_DAY,
    cameraPosition: [0, 0.2, 4.85],
    fov: 35,
    autoRotateSpeed: 0.45,
  });

  const celadon = createPigmentUniforms('celadonGlauconite');
  const calcite = createPigmentUniforms('boneCalcite');
  const umber = createPigmentUniforms('rawUmber');
  const amber = createPigmentUniforms('realgarAmber');

  // 1. Central Foliated Mineral Specimen (TorusKnot + TSL vertex foliation & Kubelka-Munk glaze)
  const coreGeom = new THREE.TorusKnotGeometry(0.76, 0.25, 220, 48, 2, 3);
  const coreMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.42,
    metalness: 0.06,
  });

  const pos = positionLocal;
  const norm = normalLocal;

  // Pointer-field proximity in local/world space
  const toPointer = pos.sub(stage.uPointerWorld);
  const pointerDist = length(toPointer);
  const pointerInfluence = exp(pointerDist.mul(pointerDist).negate().mul(1.85));

  // Organic metamorphic foliation ridges (zero polar star harmonics)
  const tooth = mineralToothFbm(pos.add(stage.uTime.mul(0.06)), float(3.4));
  const foliationWave = sin(
    dot(pos, vec3(4.2, 5.7, 3.8)).add(tooth.mul(4.5)).add(stage.uTime.mul(0.9))
  );

  // Vertex displacement: breathing mineral strata + tactile pointer shear/expansion
  const dispAmp = float(0.032)
    .add(pointerInfluence.mul(0.085))
    .add(stage.uPointerMeta.z.mul(0.045));
  coreMat.positionNode = pos.add(norm.mul(foliationWave.mul(dispAmp)));

  // Physical Kubelka-Munk glaze thickness driven by foliation ridges & pointer dwell
  const glazeThickness = clamp(
    float(0.38).add(foliationWave.mul(0.28)).add(tooth.mul(0.24)).sub(pointerInfluence.mul(0.22)),
    0.04,
    1.15
  );

  const celadonOverCalcite = kubelkaMunkGlaze(
    celadon.K,
    celadon.S,
    glazeThickness,
    calcite.rInf
  );

  // Subtle raw umber crevice darkening in deep folds
  const creviceMask = smoothstep(0.15, -0.55, foliationWave);
  const baseMineralRgb = mix(celadonOverCalcite, umber.rInf, creviceMask.mul(0.42));

  // Press-and-hold or fast pointer sweep excites warm Realgar Amber mineral seam resonance
  const amberExcite = clamp(
    pointerInfluence.mul(
      float(0.35).add(stage.uPointerMeta.y.mul(0.55)).add(stage.uPointerMeta.z.mul(0.75))
    ),
    0.0,
    0.85
  );
  coreMat.colorNode = mix(baseMineralRgb, amber.rInf, amberExcite);
  coreMat.roughnessNode = clamp(float(0.52).sub(tooth.mul(0.22)).sub(amberExcite.mul(0.15)), 0.18, 0.78);

  const coreMesh = new THREE.Mesh(coreGeom, coreMat);
  coreMesh.castShadow = true;
  coreMesh.receiveShadow = true;
  stage.specimenGroup.add(coreMesh);

  // 2. Suspended Calibration Gimbal Ring (Raw Umber & Calcite Specimen Mount)
  const ringGeom = new THREE.TorusGeometry(1.34, 0.016, 24, 160);
  const ringMat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.36,
    metalness: 0.45,
  });
  const ringTooth = mineralToothFbm(positionLocal, float(6.0));
  ringMat.colorNode = mix(umber.rInf, calcite.rInf, ringTooth.mul(0.38));

  const outerRing = new THREE.Mesh(ringGeom, ringMat);
  outerRing.rotation.x = Math.PI * 0.32;
  outerRing.rotation.y = Math.PI * 0.14;
  stage.specimenGroup.add(outerRing);

  const innerRingGeom = new THREE.TorusGeometry(1.18, 0.011, 20, 140);
  const innerRing = new THREE.Mesh(innerRingGeom, ringMat);
  innerRing.rotation.x = -Math.PI * 0.24;
  innerRing.rotation.z = Math.PI * 0.28;
  stage.specimenGroup.add(innerRing);

  // 3. Subtle rotational coupling to pointer velocity + gentle suspension float
  const rotVel = new THREE.Vector2(0, 0);
  stage.onTick(({ elapsedSec, dtSec, pointer }) => {
    rotVel.x += pointer.velocity.y * 0.0018 - rotVel.x * Math.min(1, dtSec * 4.5);
    rotVel.y += pointer.velocity.x * 0.0018 - rotVel.y * Math.min(1, dtSec * 4.5);

    coreMesh.rotation.x += dtSec * 0.16 + rotVel.x;
    coreMesh.rotation.y += dtSec * 0.22 + rotVel.y;

    outerRing.rotation.z = Math.sin(elapsedSec * 0.45) * 0.18 + rotVel.y * 0.5;
    innerRing.rotation.y = Math.cos(elapsedSec * 0.38) * 0.24 - rotVel.x * 0.5;
    stage.specimenGroup.position.y = Math.sin(elapsedSec * 1.1) * 0.035;
  });

  // Suppress unused import lint
  void max;
  void cos;

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay00Specimen();
}
