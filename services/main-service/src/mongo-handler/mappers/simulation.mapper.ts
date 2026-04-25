export type SimulationListItemDto = {
  id: string;
  createdAt: Date;
  durationSeconds: number;
  weatherSource: "machine" | "api" | "calculations";
};

export type CreateSimulationResponseDto = {
  success: true;
  simulation: SimulationListItemDto;
  access: {
    token: string;
    headerName: string;
  };
};

export type SimulationDetailsDto = {
  createdAt: Date;
  durationSeconds: number;
  weatherSource: "machine" | "api" | "calculations";
  initialData: {
    lat: number;
    lon: number;
    alt: number;
    azimuth: number;
    elevation: number;
    mass: number;
    initialSpeed: number;
    weather_source?: "machine" | "api" | "calculations";
  };
};

export type SimulationWatchDto = {
  coordinates: {
    lon: number;
    lat: number;
    alt: number;
  }[];
};

export function toSimulationListItemDto(input: {
  _id: { toString(): string };
  createdAt: Date;
  durationSeconds: number;
  weather_source: "machine" | "api" | "calculations";
}): SimulationListItemDto {
  return {
    id: input._id.toString(),
    createdAt: input.createdAt,
    durationSeconds: input.durationSeconds,
    weatherSource: input.weather_source,
  };
}
