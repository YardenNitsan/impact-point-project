import mongoose from "mongoose";
import { coordinateSchema } from "./subSchemas.model";
import { defaultSchemaOptions } from "./SchemaOptions.model";
import { initialDataSchema, InitialData } from "./initialData.model";

export interface SimulationDetailesDocument extends mongoose.Document {
  initialData: InitialData;
  coords: {
    lon: number;
    lat: number;
    alt: number;
  }[];
  durationMinutes: number;
}

const SimulationDetailesSchema = new mongoose.Schema(
  {
    initialData: {
      type: initialDataSchema,
      required: true,
    },
    coords: {
      type: [coordinateSchema],
      required: true,
    },
    durationMinutes: {
      type: Number,
      required: true,
    },
  },
  defaultSchemaOptions,
);

export const SimulationDetailes = mongoose.model<SimulationDetailesDocument>(
  "SimulationDetail",
  SimulationDetailesSchema,
);
