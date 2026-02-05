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
  durationMinutes: number;
  createdAt: Date;   // 🔥 זה היה חסר
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
    durationMinutes: {
      type: Number,
      required: true,
    },
  },
  defaultSchemaOptions
);

export const SimulationResult =
  mongoose.model<SimulationResultDocument>(
    "SimulationResult",
    SimulationResultSchema
  );
