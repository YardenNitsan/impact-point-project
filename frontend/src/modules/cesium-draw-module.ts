import {
  Cartesian3,
  Color,
  SampledPositionProperty,
  VelocityOrientationProperty,
  JulianDate,
  ClockRange,
  HermitePolynomialApproximation,
} from "cesium";

export function drawTrajectory(viewer: any, points: any[]) {
  const viewerWidthValue:number = 5;
  const interpolationDegreeValue:number = 2; 
  var totalTime:number = 0;
  const entityPixelSize: number = 14;

  const pathPositions = points.map(p =>
    Cartesian3.fromDegrees(p.lon, p.lat, p.alt)
  );

  viewer.entities.add({
    polyline: {
      positions: pathPositions,
      viewerWidth: viewerWidthValue,
      material: Color.CYAN
    }
  });

  const property = new SampledPositionProperty();
  const startTime = JulianDate.now();

  property.setInterpolationOptions({
    interpolationAlgorithm: HermitePolynomialApproximation,
    interpolationDegree: interpolationDegreeValue
  });

  for (let i = 0; i < pathPositions.length - 1; i++) {
    const start = pathPositions[i];
    const end = pathPositions[i + 1];

    for (let t = 0; t <= 1; t += 0.01) {
      const point = Cartesian3.lerp(start, end, t, new Cartesian3());
      const time = JulianDate.addSeconds(startTime, totalTime, new JulianDate());
      property.addSample(time, point);
      totalTime += 0.1;
    }
  }

  const stopTime = JulianDate.addSeconds(startTime, totalTime, new JulianDate());

  viewer.clock.startTime = startTime.clone();
  viewer.clock.stopTime = stopTime.clone();
  viewer.clock.currentTime = startTime.clone();
  viewer.clock.clockRange = ClockRange.CLAMPED;
  viewer.clock.shouldAnimate = true;

  const entity = viewer.entities.add({
    position: property,
    point: { 
      pixelSize: entityPixelSize, 
      color: Color.RED
    },
    orientation: new VelocityOrientationProperty(property)
  });

  viewer.trackedEntity = entity;
}
