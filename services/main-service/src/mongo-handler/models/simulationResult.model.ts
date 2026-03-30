import mongoose, { Types } from "mongoose";
import { coordinateSchema } from "./subSchemas.model";
import { defaultSchemaOptions } from "./SchemaOptions.model";

export interface SimulationResultDocument extends mongoose.Document {
  simulationInputId: Types.ObjectId;
  coordinates: {
    lon: number;
    lat: number;
    alt: number;
  }[];
  durationSeconds: number;
  weather_source: "machine" | "api" | "calculations";
  accessTokenHash: string;
  createdAt: Date;
}

const SimulationResultSchema = new mongoose.Schema(
  {
    simulationInputId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "SimulationInput",
      required: true,
    },
    coordinates: {
      type: [coordinateSchema],
      required: true,
      default: [],
    },
    durationSeconds: {
      type: Number,
      required: true,
    },
    weather_source: {
      type: String,
      enum: ["machine", "api", "calculations"],
      required: true,
      default: "machine",
    },
    accessTokenHash: {
      type: String,
      required: true,
      select: false,
    },
  },
  defaultSchemaOptions,
);

export const SimulationResult = mongoose.model<SimulationResultDocument>(
  "SimulationResult",
  SimulationResultSchema,
);
