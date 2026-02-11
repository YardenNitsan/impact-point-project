import {
  ArcType,
  CallbackProperty,
  Cartesian3,
  Cartographic,
  Color,
  Entity,
  ExtrapolationType,
  HeightReference,
  JulianDate,
  LagrangePolynomialApproximation,
  SampledPositionProperty,
  VelocityOrientationProperty,
  Viewer,
  sampleTerrainMostDetailed,
} from "cesium";

import { Coordinate } from "../../services/shared.service";

/* =========================
   Performance constants
   ========================= */

const TERRAIN_ALTITUDE_OFFSET_METERS = 5;

const POLYLINE_ALPHA = 0.95;
const MOVING_POINT_PIXEL_SIZE = 12;

const INTERPOLATION_DEGREE = 1;

const MAX_TERRAIN_SAMPLES = 1200;
const MAX_ANIMATION_SAMPLES = 2500;

// safe duration seconds
const DEFAULT_ANIMATION_DURATION_SECONDS = 20;

// LOD thresholds (meters)
const LOD_HEIGHT_MEDIUM = 50_000;
const LOD_HEIGHT_FAR = 200_000;

// Polyline max points by LOD
const LOD_POINTS_NEAR = 2500;
const LOD_POINTS_MEDIUM = 2000;
const LOD_POINTS_FAR = 700;

// Polyline width by LOD
const POLYLINE_WIDTH_NEAR = 4;
const POLYLINE_WIDTH_MEDIUM = 3;
const POLYLINE_WIDTH_FAR = 2;

/* =========================
   Terrain cache
   ========================= */

const TERRAIN_CACHE_MAX_KEYS = 8;
const terrainCache = new Map<string, number[]>();

function terrainCacheGet(key: string): number[] | undefined {
  const v = terrainCache.get(key);
  if (!v) return undefined;
  terrainCache.delete(key);
  terrainCache.set(key, v);
  return v;
}

function terrainCacheSet(key: string, heights: number[]) {
  if (terrainCache.has(key)) terrainCache.delete(key);
  terrainCache.set(key, heights);

  while (terrainCache.size > TERRAIN_CACHE_MAX_KEYS) {
    const firstKey = terrainCache.keys().next().value;
    if (firstKey !== undefined) terrainCache.delete(firstKey);
  }
}

/* =========================
   Types
   ========================= */

export type TrajectoryLODHandles = {
  rawPoints: Coordinate[];

  fullPositions?: Cartesian3[];

  // polyline
  polylinePositions?: Cartesian3[];
  polylinePositionsCallback?: CallbackProperty;

  polylineWidth?: number;
  polylineWidthCallback?: CallbackProperty;

  polylineEntity?: Entity;

  // moving entity
  movingProperty?: SampledPositionProperty;
  movingEntity?: Entity;

  lastKey?: string;
  lastDurationSeconds?: number;
};

/* =========================
   Helpers
   ========================= */

function altitudeWithOffsetMeters(alt: number) {
  return Math.max(0, (alt ?? 0) + TERRAIN_ALTITUDE_OFFSET_METERS);
}

function buildTrajectoryKey(points: Coordinate[]): string {
  const n = points.length;
  if (n === 0) return "empty";

  const a = points[0];
  const b = points[Math.floor(n / 2)];
  const c = points[n - 1];

  const pack = (p: Coordinate) =>
    `${p.lon.toFixed(6)}|${p.lat.toFixed(6)}|${(p.alt ?? 0).toFixed(2)}`;

  return `${n}::${pack(a)}::${pack(b)}::${pack(c)}`;
}

function getLODMaxPoints(viewer: Viewer): number {
  const h = viewer.camera.positionCartographic.height;
  if (h > LOD_HEIGHT_FAR) return LOD_POINTS_FAR;
  if (h > LOD_HEIGHT_MEDIUM) return LOD_POINTS_MEDIUM;
  return LOD_POINTS_NEAR;
}

function getLODWidth(viewer: Viewer): number {
  const h = viewer.camera.positionCartographic.height;
  if (h > LOD_HEIGHT_FAR) return POLYLINE_WIDTH_FAR;
  if (h > LOD_HEIGHT_MEDIUM) return POLYLINE_WIDTH_MEDIUM;
  return POLYLINE_WIDTH_NEAR;
}

function fillDownsampled(out: Cartesian3[], full: Cartesian3[], maxPoints: number) {
  out.length = 0;

  const n = full.length;
  if (n === 0) return;

  if (n <= maxPoints) {
    for (let i = 0; i < n; i++) out.push(full[i]);
    return;
  }

  const step = Math.max(1, Math.ceil(n / maxPoints));

  for (let i = 0; i < n; i += step) out.push(full[i]);
  if (out[out.length - 1] !== full[n - 1]) out.push(full[n - 1]);
}

function normalizeDurationSeconds(durationSeconds: number): number {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return DEFAULT_ANIMATION_DURATION_SECONDS;
  }
  return durationSeconds;
}

/* =========================
   Terrain sampling (cached)
   ========================= */

async function sampleTerrainFastCached(
  viewer: Viewer,
  points: Coordinate[],
  key: string
): Promise<number[]> {
  const cached = terrainCacheGet(key);
  if (cached && cached.length === points.length) return cached;

  const n = points.length;
  const heights = new Array<number>(n).fill(0);

  if (n === 0) {
    terrainCacheSet(key, heights);
    return heights;
  }

  const sampleStep = Math.max(1, Math.ceil(n / MAX_TERRAIN_SAMPLES));

  const indices: number[] = [];
  const cartos: Cartographic[] = [];

  for (let i = 0; i < n; i += sampleStep) {
    indices.push(i);
    const p = points[i];
    cartos.push(Cartographic.fromDegrees(p.lon, p.lat));
  }

  if (indices[indices.length - 1] !== n - 1) {
    indices.push(n - 1);
    const p = points[n - 1];
    cartos.push(Cartographic.fromDegrees(p.lon, p.lat));
  }

  let updated: Cartographic[];
  try {
    updated = await sampleTerrainMostDetailed(viewer.terrainProvider, cartos);
  } catch {
    terrainCacheSet(key, heights);
    return heights;
  }

  for (let i = 0; i < indices.length; i++) {
    const h = updated[i]?.height;
    heights[indices[i]] = Number.isFinite(h) ? (h as number) : 0;
  }

  // interpolate between sampled points
  for (let s = 0; s < indices.length - 1; s++) {
    const a = indices[s];
    const b = indices[s + 1];

    const ha = heights[a];
    const hb = heights[b];

    const span = b - a;
    if (span <= 1) continue;

    for (let i = a + 1; i < b; i++) {
      const t = (i - a) / span;
      heights[i] = ha + (hb - ha) * t;
    }
  }

  terrainCacheSet(key, heights);
  return heights;
}

/* =========================
   LOD update
   ========================= */

export function updateTrajectoryLOD(viewer: Viewer, handles: TrajectoryLODHandles) {
  if (!handles.fullPositions || !handles.polylinePositions || !handles.polylineEntity) return;

  const maxPoints = getLODMaxPoints(viewer);
  const width = getLODWidth(viewer);

  fillDownsampled(handles.polylinePositions, handles.fullPositions, maxPoints);
  handles.polylineWidth = width;

  viewer.scene.requestRender();
}

/* =========================
   Main draw
   ========================= */

export async function drawTrajectoryLOD(
  viewer: Viewer,
  rawPoints: Coordinate[],
  handles: TrajectoryLODHandles,
  durationSeconds: number
): Promise<TrajectoryLODHandles> {
  if (!viewer || rawPoints.length === 0) return handles;

  handles.rawPoints = rawPoints;

  const key = buildTrajectoryKey(rawPoints);
  const safeDuration = normalizeDurationSeconds(durationSeconds);

  const sameTrajectory = handles.lastKey === key;
  const sameDuration = handles.lastDurationSeconds === safeDuration;

  if (sameTrajectory && sameDuration && handles.fullPositions) {
    updateTrajectoryLOD(viewer, handles);
    return handles;
  }

  handles.lastKey = key;
  handles.lastDurationSeconds = safeDuration;

  // terrain
  const terrain = await sampleTerrainFastCached(viewer, rawPoints, key);

  // full positions
  const full = new Array<Cartesian3>(rawPoints.length);
  for (let i = 0; i < rawPoints.length; i++) {
    const p = rawPoints[i];
    const h = altitudeWithOffsetMeters(p.alt) + (terrain[i] ?? 0);
    full[i] = Cartesian3.fromDegrees(p.lon, p.lat, Number.isFinite(h) ? h : 0);
  }
  handles.fullPositions = full;

  // polyline reusable (CallbackProperty for positions + width)
  if (!handles.polylinePositions) handles.polylinePositions = [];
  if (handles.polylineWidth === undefined) handles.polylineWidth = POLYLINE_WIDTH_NEAR;

  if (!handles.polylinePositionsCallback) {
    handles.polylinePositionsCallback = new CallbackProperty(() => {
      return handles.polylinePositions!;
    }, false);
  }

  if (!handles.polylineWidthCallback) {
    handles.polylineWidthCallback = new CallbackProperty(() => {
      return handles.polylineWidth ?? POLYLINE_WIDTH_NEAR;
    }, false);
  }

  if (!handles.polylineEntity) {
    handles.polylineEntity = viewer.entities.add({
      polyline: {
        positions: handles.polylinePositionsCallback,
        width: handles.polylineWidthCallback,
        material: Color.CYAN.withAlpha(POLYLINE_ALPHA),
        arcType: ArcType.GEODESIC,
      },
    });
  }

  // fill initial LOD
  updateTrajectoryLOD(viewer, handles);

  // moving entity
  handles.movingProperty = new SampledPositionProperty();
  handles.movingProperty.setInterpolationOptions({
    interpolationAlgorithm: LagrangePolynomialApproximation,
    interpolationDegree: INTERPOLATION_DEGREE,
  });
  handles.movingProperty.forwardExtrapolationType = ExtrapolationType.HOLD;
  handles.movingProperty.backwardExtrapolationType = ExtrapolationType.HOLD;

  const n = full.length;
  const step = Math.max(1, Math.ceil(n / MAX_ANIMATION_SAMPLES));

  const start = JulianDate.now();
  const dt = safeDuration / Math.max(1, n - 1);

  for (let i = 0; i < n; i += step) {
    const t = JulianDate.addSeconds(start, i * dt, new JulianDate());
    handles.movingProperty.addSample(t, full[i]);
  }

  if ((n - 1) % step !== 0) {
    const t = JulianDate.addSeconds(start, (n - 1) * dt, new JulianDate());
    handles.movingProperty.addSample(t, full[n - 1]);
  }

  if (!handles.movingEntity) {
    handles.movingEntity = viewer.entities.add({
      position: handles.movingProperty,
      point: {
        pixelSize: MOVING_POINT_PIXEL_SIZE,
        color: Color.RED,
        heightReference: HeightReference.NONE,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      orientation: new VelocityOrientationProperty(handles.movingProperty),
    });
  } else {
    handles.movingEntity.position = handles.movingProperty;
    handles.movingEntity.orientation = new VelocityOrientationProperty(handles.movingProperty);
  }

  viewer.clock.shouldAnimate = true;
  viewer.trackedEntity = handles.movingEntity;

  viewer.scene.requestRender();
  return handles;
}
