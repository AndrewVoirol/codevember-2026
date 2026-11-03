#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const THEMES = [
  null,
  { day: 1, title: 'Fold', substrate: 'tsl', primary: 'boneCalcite', secondary: 'rawUmber' },
  { day: 2, title: 'Shatter', substrate: 'rapier', primary: 'azuritePrussian', secondary: 'realgarAmber' },
  { day: 3, title: 'Melt', substrate: 'tsl-compute', primary: 'realgarAmber', secondary: 'burntSienna' },
  { day: 4, title: 'Tangle', substrate: 'tsl-compute', primary: 'burntSienna', secondary: 'boneCalcite' },
  { day: 5, title: 'Moss', substrate: 'tsl', primary: 'mossUmbrous', secondary: 'yellowOchre' },
  { day: 6, title: 'Orrery', substrate: 'tsl', primary: 'yellowOchre', secondary: 'lapisUltramarine' },
  { day: 7, title: 'Hollow', substrate: 'tsl', primary: 'warmKaolin', secondary: 'cinnabarVermilion' },
  { day: 8, title: 'Tesseract', substrate: 'tsl', primary: 'lapisUltramarine', secondary: 'boneCalcite' },
  { day: 9, title: 'Membrane', substrate: 'tsl-compute', primary: 'celadonGlauconite', secondary: 'cinnabarVermilion' },
  { day: 10, title: 'Caustic', substrate: 'wgsl', primary: 'celadonGlauconite', secondary: 'realgarAmber' },
  { day: 11, title: 'Rust', substrate: 'tsl-compute', primary: 'hematiteRust', secondary: 'verdigrisMalachite' },
  { day: 12, title: 'Feather', substrate: 'tsl', primary: 'boneCalcite', secondary: 'azuritePrussian' },
  { day: 13, title: 'Labyrinth', substrate: 'tsl', primary: 'rawUmber', secondary: 'yellowOchre' },
  { day: 14, title: 'Tension', substrate: 'tsl-compute', primary: 'cinnabarVermilion', secondary: 'boneCalcite' },
  { day: 15, title: 'Mirage', substrate: 'wgsl', primary: 'realgarAmber', secondary: 'lapisUltramarine' },
  { day: 16, title: 'Spore', substrate: 'tsl-compute', primary: 'sporeSulfur', secondary: 'rawUmber' },
  { day: 17, title: 'Kelp', substrate: 'tsl-compute', primary: 'mossUmbrous', secondary: 'realgarAmber' },
  { day: 18, title: 'Pendulum', substrate: 'tsl', primary: 'boneCalcite', secondary: 'burntSienna' },
  { day: 19, title: 'Erosion', substrate: 'wgsl', primary: 'yellowOchre', secondary: 'burntSienna' },
  { day: 20, title: 'Cocoon', substrate: 'tsl', primary: 'warmKaolin', secondary: 'realgarAmber' },
  { day: 21, title: 'Bolt', substrate: 'wgsl', primary: 'boneCalcite', secondary: 'azuritePrussian' },
  { day: 22, title: 'Echo', substrate: 'wgsl', primary: 'verdigrisMalachite', secondary: 'boneCalcite' },
  { day: 23, title: 'Magma', substrate: 'wgsl', primary: 'cinnabarVermilion', secondary: 'rawUmber' },
  { day: 24, title: 'Nest', substrate: 'tsl', primary: 'rawUmber', secondary: 'celadonGlauconite' },
  { day: 25, title: 'Veil', substrate: 'tsl-compute', primary: 'boneCalcite', secondary: 'lapisUltramarine' },
  { day: 26, title: 'Anamorph', substrate: 'tsl', primary: 'yellowOchre', secondary: 'azuritePrussian' },
  { day: 27, title: 'Burrow', substrate: 'tsl', primary: 'burntSienna', secondary: 'boneCalcite' },
  { day: 28, title: 'Hive', substrate: 'tsl', primary: 'yellowOchre', secondary: 'realgarAmber' },
  { day: 29, title: 'Portal', substrate: 'tsl', primary: 'lapisUltramarine', secondary: 'verdigrisMalachite' },
  { day: 30, title: 'Dissolve', substrate: 'tsl-compute', primary: 'boneCalcite', secondary: 'celadonGlauconite' },
];

function buildHtmlTemplate(pad, title, slug) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" href="data:," />
    <title>Day ${pad} - ${title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400&display=swap"
      rel="stylesheet"
    />
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/days/${slug}/main.ts"></script>
  </body>
</html>
`;
}

function buildTslMainTemplate(day, pad, title, primary, secondary) {
  return `import * as THREE from 'three/webgpu';
import {
  clamp,
  dot,
  exp,
  float,
  length,
  mix,
  normalLocal,
  positionLocal,
  sin,
  vec3,
} from 'three/tsl';
import { getDayByNumber } from '../../core/manifest';
import {
  createPigmentUniforms,
  kubelkaMunkGlaze,
  mineralToothFbm,
} from '../../core/tslMaterials';
import { SpecimenStage } from '../../core/stage/SpecimenStage';

export async function bootDay${pad}${title}(): Promise<SpecimenStage> {
  const dayEntry = getDayByNumber(${day});
  const stage = await SpecimenStage.create({
    day: dayEntry,
    cameraPosition: [0, 0.3, 4.2],
    fov: 35,
  });

  const primaryPigment = createPigmentUniforms('${primary}');
  const secondaryPigment = createPigmentUniforms('${secondary}');

  const geom = new THREE.IcosahedronGeometry(0.95, 5);
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.45,
    metalness: 0.05,
  });

  const pos = positionLocal;
  const norm = normalLocal;
  const toPointer = pos.sub(stage.uPointerWorld);
  const pointerInfluence = exp(length(toPointer).mul(length(toPointer)).negate().mul(1.8));

  const tooth = mineralToothFbm(pos.add(stage.uTime.mul(0.05)), float(3.2));
  const wave = sin(dot(pos, vec3(3.8, 4.6, 3.2)).add(tooth.mul(4.0)).add(stage.uTime));

  mat.positionNode = pos.add(norm.mul(wave.mul(float(0.03).add(pointerInfluence.mul(0.08)))));

  const thickness = clamp(float(0.42).add(wave.mul(0.25)).sub(pointerInfluence.mul(0.2)), 0.05, 1.1);
  const glazed = kubelkaMunkGlaze(
    primaryPigment.K,
    primaryPigment.S,
    thickness,
    secondaryPigment.rInf
  );

  mat.colorNode = mix(glazed, secondaryPigment.rInf, pointerInfluence.mul(0.45));

  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  stage.specimenGroup.add(mesh);

  stage.onTick(({ dtSec, pointer }) => {
    mesh.rotation.y += dtSec * 0.18 + pointer.velocity.x * 0.0015;
    mesh.rotation.x += dtSec * 0.11 + pointer.velocity.y * 0.0015;
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay${pad}${title}();
}
`;
}

function buildWgslMainTemplate(day, pad, title) {
  return `import { getDayByNumber } from '../../core/manifest';
import { RawWebGPUStage } from '../../core/stage/RawWebGPUStage';

const SHADER_WGSL = /* wgsl */ \`
struct Uniforms {
  resolution: vec2<f32>,
  timeSec: f32,
  dtSec: f32,
  pointerNdc: vec2<f32>,
  pointerDwell: f32,
  pointerImpulse: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  let p = positions[vid];
  var out: VsOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = p * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let aspect = u.resolution.x / max(1.0, u.resolution.y);
  let p = (in.uv * 2.0 - 1.0) * vec2<f32>(aspect, 1.0);
  let ptr = u.pointerNdc * vec2<f32>(aspect, 1.0);
  let d = length(p - ptr);
  let slate = vec3<f32>(0.0152, 0.0185, 0.0222);
  let calcite = vec3<f32>(0.896, 0.863, 0.791);
  let ring = exp(-pow(abs(length(p) - 0.42 - 0.04 * sin(u.timeSec + d * 6.0)), 2.0) * 180.0);
  let col = mix(slate, calcite, clamp(ring, 0.0, 1.0));
  return vec4<f32>(col, 1.0);
}
\`;

export async function bootDay${pad}${title}(): Promise<RawWebGPUStage> {
  const dayEntry = getDayByNumber(${day});
  const stage = await RawWebGPUStage.create({ day: dayEntry });
  const { device, format } = stage;

  const uniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformData = new Float32Array(8);

  const module = device.createShaderModule({ code: SHADER_WGSL });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_main' },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  stage.onTick((ctx) => {
    uniformData[0] = ctx.width;
    uniformData[1] = ctx.height;
    uniformData[2] = ctx.elapsedSec;
    uniformData[3] = ctx.dtSec;
    uniformData[4] = ctx.pointerNdc[0];
    uniformData[5] = ctx.pointerNdc[1];
    uniformData[6] = ctx.pointerDwellSec;
    uniformData[7] = ctx.pointerImpulse;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.context.getCurrentTexture().createView(),
          clearValue: stage.clearColor,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  });

  return stage;
}

if (typeof document !== 'undefined' && document.getElementById('app')) {
  void bootDay${pad}${title}();
}
`;
}

function main() {
  const args = process.argv.slice(2);
  const dayArg = args.find((a) => !a.startsWith('--'));
  if (!dayArg) {
    console.error('Usage: npm run day:new <01..30> [--substrate=tsl|tsl-compute|rapier|wgsl] [--dry-run] [--force]');
    process.exit(1);
  }

  const dayNum = Number.parseInt(dayArg, 10);
  if (!Number.isInteger(dayNum) || dayNum < 1 || dayNum > 30) {
    console.error(`Invalid day "${dayArg}". Expected an integer between 1 and 30.`);
    process.exit(1);
  }

  const meta = THEMES[dayNum];
  const pad = String(dayNum).padStart(2, '0');
  const slug = `${pad}-${meta.title.toLowerCase()}`;
  const substrateFlag = args.find((a) => a.startsWith('--substrate='))?.split('=')[1];
  const substrate = substrateFlag || meta.substrate;
  const dryRun = args.includes('--dry-run');
  const force = args.includes('--force');

  const htmlRelPath = `${slug}/index.html`;
  const mainRelPath = `src/days/${slug}/main.ts`;
  const htmlAbsPath = path.resolve(ROOT_DIR, htmlRelPath);
  const mainAbsPath = path.resolve(ROOT_DIR, mainRelPath);

  const htmlContent = buildHtmlTemplate(pad, meta.title, slug);
  const mainContent =
    substrate === 'wgsl'
      ? buildWgslMainTemplate(dayNum, pad, meta.title)
      : buildTslMainTemplate(dayNum, pad, meta.title, meta.primary, meta.secondary);

  if (dryRun) {
    console.log(
      JSON.stringify(
        {
          day: dayNum,
          slug,
          title: meta.title,
          substrate,
          files: [htmlRelPath, mainRelPath],
        },
        null,
        2
      )
    );
    return;
  }

  if (!force && (fs.existsSync(htmlAbsPath) || fs.existsSync(mainAbsPath))) {
    console.error(`Files for ${slug} already exist. Pass --force to overwrite.`);
    process.exit(1);
  }

  fs.mkdirSync(path.dirname(htmlAbsPath), { recursive: true });
  fs.mkdirSync(path.dirname(mainAbsPath), { recursive: true });
  fs.writeFileSync(htmlAbsPath, htmlContent, 'utf8');
  fs.writeFileSync(mainAbsPath, mainContent, 'utf8');

  console.log(`Scaffolded Day ${pad} - ${meta.title} (${substrate}):`);
  console.log(`  - ${htmlRelPath}`);
  console.log(`  - ${mainRelPath}`);
}

main();
