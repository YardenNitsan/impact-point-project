export interface CreateSimulationBody {
  alt: number;
  azimuth: number;
  elevation: number;
  lat: number;
  lon: number;
  mass: number;
  initialSpeed: number;
  weather_source?: "machine" | "api" | "knn" | "calculations";
}
