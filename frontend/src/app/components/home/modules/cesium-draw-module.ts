import {
  CallbackProperty,
  Cartesian3,
  Cartographic,
  Color,
  Entity,
  ExtrapolationType,
  HeightReference,
  JulianDate,
  LinearApproximation,
  SampledPositionProperty,
  VelocityOrientationProperty,
  Viewer,
  sampleTerrainMostDetailed,
  Cartesian2,
  HorizontalOrigin,
  LabelStyle,
  Math as CesiumMath,
  NearFarScalar,
  VerticalOrigin,
  LabelGraphics,
  ClockRange,
} from 'cesium';

import { Coordinate } from '../../models/coordinate.model';

/* =========================
   Performance constants
   ========================= */

const POLYLINE_ALPHA = 0.95;
const MOVING_POINT_PIXEL_SIZE = 12;

const MAX_TERRAIN_SAMPLES = 300;
const MAX_ANIMATION_SAMPLES = 2500;

// safe duration seconds
const DEFAULT_ANIMATION_DURATION_SECONDS = 20;

// LOD thresholds (meters)
const CAMERA_HEIGHT_LOD_MEDIUM_METERS = 50_000;
const CAMERA_HEIGHT_LOD_FAR_METERS = 200_000;

// Polyline max points by LOD
const MAX_POLYLINE_POINTS_LOD_NEAR = 300;
const MAX_POLYLINE_POINTS_LOD_MEDIUM = 200;
const MAX_POLYLINE_POINTS_LOD_FAR = 120;

// Polyline width by LOD
const POLYLINE_WIDTH_NEAR = 4;
const POLYLINE_WIDTH_MEDIUM = 3;
const POLYLINE_WIDTH_FAR = 2;

// watch simulation speed
const DEFAULT_SIMULATION_SPEED = 1;

// Label Constants
const LABEL_SCALE_NEAR_DISTANCE = 2000;
const LABEL_SCALE_NEAR_VALUE = 1.0;
const LABEL_SCALE_FAR_DISTANCE = 150000;
const LABEL_SCALE_FAR_VALUE = 0.0;

const LABEL_PIXEL_OFFSET_X = 12;
const LABEL_PIXEL_OFFSET_Y = -12;

/* =========================
   Terrain cache
   ========================= */

const TERRAIN_CACHE_MAX_KEYS = 32;
const terrainCache = new Map<string, number[]>();

function terrainCacheGet(key: string): number[] | undefined {
  const simulation_path = terrainCache.get(key);
  if (!simulation_path) return undefined;
  terrainCache.delete(key);
  terrainCache.set(key, simulation_path);
  return simulation_path;
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

function buildTrajectoryKey(points: Coordinate[]): string {
  const totalPoints = points.length;
  if (totalPoints === 0) return 'empty';

  const first_point = points[0];
  const middle_point = points[Math.floor(totalPoints / 2)];
  const last_point = points[totalPoints - 1];

  const pack = (p: Coordinate) =>
    `${p.lon.toFixed(6)}|${p.lat.toFixed(6)}|${(p.alt ?? 0).toFixed(2)}`;

  return `${totalPoints}::${pack(first_point)}::${pack(middle_point)}::${pack(last_point)}`;
}

function getLODMaxPoints(viewer: Viewer): number {
  const camera_height_above_ground = viewer.camera.positionCartographic.height;
  if (camera_height_above_ground > CAMERA_HEIGHT_LOD_FAR_METERS)
    return MAX_POLYLINE_POINTS_LOD_FAR;
  if (camera_height_above_ground > CAMERA_HEIGHT_LOD_MEDIUM_METERS)
    return MAX_POLYLINE_POINTS_LOD_MEDIUM;
  return MAX_POLYLINE_POINTS_LOD_NEAR;
}

function getLODWidth(viewer: Viewer): number {
  const h = viewer.camera.positionCartographic.height;
  if (h > CAMERA_HEIGHT_LOD_FAR_METERS) return POLYLINE_WIDTH_FAR;
  if (h > CAMERA_HEIGHT_LOD_MEDIUM_METERS) return POLYLINE_WIDTH_MEDIUM;
  return POLYLINE_WIDTH_NEAR;
}

function fillDownsampled(
  downsampledPositions: Cartesian3[],
  fullPositions: Cartesian3[],
  maxAllowedPoints: number,
) {
  downsampledPositions.length = 0;

  const totalFullPoints = fullPositions.length;
  if (totalFullPoints === 0) return;

  if (totalFullPoints <= maxAllowedPoints) {
    for (let i = 0; i < totalFullPoints; i++)
      downsampledPositions.push(fullPositions[i]);
    return;
  }

  const step = Math.max(1, Math.ceil(totalFullPoints / maxAllowedPoints));

  for (let i = 0; i < totalFullPoints; i += step)
    downsampledPositions.push(fullPositions[i]);
  if (
    downsampledPositions[downsampledPositions.length - 1] !==
    fullPositions[totalFullPoints - 1]
  )
    downsampledPositions.push(fullPositions[totalFullPoints - 1]);
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
  key: string,
): Promise<number[]> {
  const cached = terrainCacheGet(key);
  if (cached && cached.length === points.length) return cached;

  const totalPoints = points.length;
  const terrainHeightsMeters = new Array<number>(totalPoints).fill(0);

  if (totalPoints === 0) {
    terrainCacheSet(key, terrainHeightsMeters);
    return terrainHeightsMeters;
  }

  const sampleStep = Math.max(1, Math.ceil(totalPoints / MAX_TERRAIN_SAMPLES));

  const sampledPointIndices: number[] = [];
  const sampledCartographicPositions: Cartographic[] = [];

  for (let i = 0; i < totalPoints; i += sampleStep) {
    sampledPointIndices.push(i);
    const point = points[i];
    sampledCartographicPositions.push(
      Cartographic.fromDegrees(point.lon, point.lat),
    );
  }

  if (sampledPointIndices[sampledPointIndices.length - 1] !== totalPoints - 1) {
    sampledPointIndices.push(totalPoints - 1);
    const point = points[totalPoints - 1];
    sampledCartographicPositions.push(
      Cartographic.fromDegrees(point.lon, point.lat),
    );
  }

  let sampledTerrainResults: Cartographic[];
  try {
    sampledTerrainResults = await sampleTerrainMostDetailed(
      viewer.terrainProvider,
      sampledCartographicPositions,
    );
  } catch {
    terrainCacheSet(key, terrainHeightsMeters);
    return terrainHeightsMeters;
  }

  for (let i = 0; i < sampledPointIndices.length; i++) {
    const height = sampledTerrainResults[i]?.height;
    terrainHeightsMeters[sampledPointIndices[i]] = Number.isFinite(height)
      ? (height as number)
      : 0;
  }

  // interpolate between sampled points
  for (
    let segmentIndex = 0;
    segmentIndex < sampledPointIndices.length - 1;
    segmentIndex++
  ) {
    const startSampleIndex = sampledPointIndices[segmentIndex];
    const endSampleIndex = sampledPointIndices[segmentIndex + 1];

    const startHeightMeters = terrainHeightsMeters[startSampleIndex];
    const endHeightMeters = terrainHeightsMeters[endSampleIndex];

    const missing_points_amount = endSampleIndex - startSampleIndex;
    if (missing_points_amount <= 1) continue;

    for (
      let interpolatedIndex = startSampleIndex + 1;
      interpolatedIndex < endSampleIndex;
      interpolatedIndex++
    ) {
      const interpolationRatio =
        (interpolatedIndex - startSampleIndex) / missing_points_amount;
      terrainHeightsMeters[interpolatedIndex] =
        startHeightMeters +
        (endHeightMeters - startHeightMeters) * interpolationRatio;
    }
  }

  terrainCacheSet(key, terrainHeightsMeters);
  return terrainHeightsMeters;
}

/* =========================
   LOD update
   ========================= */

export function updateTrajectoryLOD(
  viewer: Viewer,
  handles: TrajectoryLODHandles,
) {
  if (
    !handles.fullPositions ||
    !handles.polylinePositions ||
    !handles.polylineEntity
  )
    return;

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
  durationSeconds: number,
): Promise<TrajectoryLODHandles> {
  if (!viewer || rawPoints.length === 0) return handles;

  handles.rawPoints = rawPoints;

  const trajectoryCacheKey = buildTrajectoryKey(rawPoints);
  const safeDuration = normalizeDurationSeconds(durationSeconds);

  const sameTrajectory = handles.lastKey === trajectoryCacheKey;
  const sameDuration = handles.lastDurationSeconds === safeDuration;

  if (sameTrajectory && sameDuration && handles.fullPositions) {
    updateTrajectoryLOD(viewer, handles);
    return handles;
  }

  handles.lastKey = trajectoryCacheKey;
  handles.lastDurationSeconds = safeDuration;

  // terrain
  const terrainHeightsMeters = await sampleTerrainFastCached(
    viewer,
    rawPoints,
    trajectoryCacheKey,
  );

  // full positions
  const fullTrajectoryPositions: Cartesian3[] = [];
  fullTrajectoryPositions.length = rawPoints.length;
  for (let i = 0; i < rawPoints.length; i++) {
    const point = rawPoints[i];
    const terrain = terrainHeightsMeters[i] ?? 0;
    const height = terrain + (point.alt ?? 0);
    fullTrajectoryPositions[i] = Cartesian3.fromDegrees(
      point.lon,
      point.lat,
      Number.isFinite(height) ? height : 0,
    );
  }
  handles.fullPositions = fullTrajectoryPositions;

  // polyline reusable (CallbackProperty for positions + width)
  if (!handles.polylinePositions) handles.polylinePositions = [];
  if (handles.polylineWidth === undefined)
    handles.polylineWidth = POLYLINE_WIDTH_NEAR;

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
      },
    });
  }

  // fill initial LOD
  updateTrajectoryLOD(viewer, handles);

  // moving entity
  handles.movingProperty = new SampledPositionProperty();
  handles.movingProperty.setInterpolationOptions({
    interpolationAlgorithm: LinearApproximation,
  });
  handles.movingProperty.forwardExtrapolationType = ExtrapolationType.HOLD;
  handles.movingProperty.backwardExtrapolationType = ExtrapolationType.HOLD;

  const totalTrajectoryPoints = fullTrajectoryPositions.length;
  const animationSampleStride = Math.max(
    1,
    Math.ceil(totalTrajectoryPoints / MAX_ANIMATION_SAMPLES),
  );

  const animationStartTime = JulianDate.fromDate(new Date(0));

  const secondsPerSample =
    safeDuration / Math.max(1, totalTrajectoryPoints - 1);

  for (let i = 0; i < totalTrajectoryPoints; i += animationSampleStride) {
    const t = JulianDate.addSeconds(
      animationStartTime,
      i * secondsPerSample,
      new JulianDate(),
    );
    handles.movingProperty.addSample(t, fullTrajectoryPositions[i]);
  }

  if ((totalTrajectoryPoints - 1) % animationSampleStride !== 0) {
    const t = JulianDate.addSeconds(
      animationStartTime,
      (totalTrajectoryPoints - 1) * secondsPerSample,
      new JulianDate(),
    );
    handles.movingProperty.addSample(
      t,
      fullTrajectoryPositions[totalTrajectoryPoints - 1],
    );
  }

  const movingLabelText = new CallbackProperty((time) => {
    const pos = handles.movingProperty?.getValue(time);
    if (!pos) return '';

    const carto = Cartographic.fromCartesian(pos);
    const lat = CesiumMath.toDegrees(carto.latitude);
    const lon = CesiumMath.toDegrees(carto.longitude);
    const terrain = viewer.scene.globe.getHeight(carto) ?? 0;
    const alt = carto.height - terrain;

    return `lat: ${lat.toFixed(6)}\nlon: ${lon.toFixed(6)}\nalt: ${alt.toFixed(1)} m`;
  }, false);

  const movingLabel = new LabelGraphics({
    text: movingLabelText,
    font: '14px monospace',
    fillColor: Color.WHITE,
    outlineColor: Color.BLACK,
    outlineWidth: 3,
    style: LabelStyle.FILL_AND_OUTLINE,
    verticalOrigin: VerticalOrigin.BOTTOM,
    horizontalOrigin: HorizontalOrigin.LEFT,
    pixelOffset: new Cartesian2(LABEL_PIXEL_OFFSET_X, LABEL_PIXEL_OFFSET_Y),
    showBackground: true,
    backgroundColor: Color.BLACK.withAlpha(0.55),
    disableDepthTestDistance: Number.POSITIVE_INFINITY,
    scaleByDistance: new NearFarScalar(
      LABEL_SCALE_NEAR_DISTANCE,
      LABEL_SCALE_NEAR_VALUE,
      LABEL_SCALE_FAR_DISTANCE,
      LABEL_SCALE_FAR_VALUE,
    ),
  });

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
      label: movingLabel,
    });
  } else {
    handles.movingEntity.position = handles.movingProperty;
    handles.movingEntity.orientation = new VelocityOrientationProperty(
      handles.movingProperty,
    );
    handles.movingEntity.label = movingLabel;
  }

  const animationStopTime = JulianDate.addSeconds(
    animationStartTime,
    safeDuration,
    new JulianDate(),
  );

  viewer.clock.startTime = animationStartTime.clone();
  viewer.clock.stopTime = animationStopTime.clone();
  viewer.clock.currentTime = animationStartTime.clone();

  viewer.clock.clockRange = ClockRange.CLAMPED;
  viewer.clock.multiplier = DEFAULT_SIMULATION_SPEED;
  viewer.timeline.zoomTo(animationStartTime, animationStopTime);
  viewer.clock.shouldAnimate = true;
  viewer.trackedEntity = handles.movingEntity;

  viewer.scene.requestRenderMode && viewer.scene.requestRender();
  return handles;
}
