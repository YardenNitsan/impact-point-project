import {
  Cartesian3,
  Color,
  SampledPositionProperty,
  VelocityOrientationProperty,
  JulianDate,
  ClockRange,
  HermitePolynomialApproximation,
  Entity,
  Viewer
} from "cesium";
import { Coordinate } from "../../services/shared.service";



export function drawTrajectory(viewer: Viewer, points: Coordinate[]) {
  if (!viewer || !points || points.length === 0) return;

  viewer.entities.removeAll();
  if (!Array.isArray(points)) {
    console.error('drawTrajectory expected array but got:', points);
    return;
  }

  if (points.length === 0) {
    console.warn('drawTrajectory got empty array');
    return;
  }

  const polylineWidth = 5;
  const interpolationDegree = 2;
  const entityPixelSize = 14;
  let totalTime = 0;

  const startTime = JulianDate.now();

  const pathPositions = points.map(p =>
    Cartesian3.fromDegrees(p.lon, p.lat, p.alt)
  );

  viewer.entities.add({
    polyline: {
      positions: pathPositions,
      width: polylineWidth,
      material: Color.CYAN
    }
  });

  const positionProperty = new SampledPositionProperty();

  positionProperty.setInterpolationOptions({
    interpolationAlgorithm: HermitePolynomialApproximation,
    interpolationDegree
  });

  for (let i = 0; i < pathPositions.length - 1; i++) {
    const start = pathPositions[i];
    const end = pathPositions[i + 1];

    for (let t = 0; t <= 1; t += 0.01) {
      const position = Cartesian3.lerp(start, end, t, new Cartesian3());
      const time = JulianDate.addSeconds(startTime, totalTime, new JulianDate());

      positionProperty.addSample(time, position);
      totalTime += 0.1;
    }
  }

  const stopTime = JulianDate.addSeconds(startTime, totalTime, new JulianDate());

  viewer.clock.startTime = startTime.clone();
  viewer.clock.stopTime = stopTime.clone();
  viewer.clock.currentTime = startTime.clone();
  viewer.clock.clockRange = ClockRange.CLAMPED;
  viewer.clock.shouldAnimate = true;


  const entity: Entity = viewer.entities.add({
    position: positionProperty,
    point: {
      pixelSize: entityPixelSize,
      color: Color.RED
    },
    orientation: new VelocityOrientationProperty(positionProperty)
  });

  viewer.flyTo(entity, {
    duration: 2
  });

  viewer.trackedEntity = entity;
}
