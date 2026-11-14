import { createLabyrinthTopology } from './labyrinthTopology';
import { buildVaultOctantRawBuffers } from './labyrinthGeometry';

interface WorkerScopeLike {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage: (message: unknown, transfer: Transferable[]) => void;
}

const scope = self as unknown as WorkerScopeLike;

scope.onmessage = () => {
  const topology = createLabyrinthTopology();
  const payload = buildVaultOctantRawBuffers(topology);
  const transfers: Transferable[] = [];
  for (let i = 0; i < payload.octants.length; i++) {
    const o = payload.octants[i];
    transfers.push(
      o.positions.buffer,
      o.normals.buffer,
      o.uvs.buffer,
      o.materialZone.buffer,
      o.stoneDetail.buffer
    );
  }
  transfers.push(
    payload.sanctumPosZ.positions.buffer,
    payload.sanctumPosZ.normals.buffer,
    payload.sanctumPosZ.sanctumZone.buffer,
    payload.sanctumNegZ.positions.buffer,
    payload.sanctumNegZ.normals.buffer,
    payload.sanctumNegZ.sanctumZone.buffer
  );
  scope.postMessage({ type: 'ready', payload }, transfers);
};
