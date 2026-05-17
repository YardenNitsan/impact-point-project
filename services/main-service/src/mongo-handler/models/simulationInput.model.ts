import mongoose from "mongoose";
import { defaultSchemaOptions } from "./SchemaOptions.model";
import { initialDataSchema, InitialData } from "./initialData.model";

export interface SimulationInputDocument extends mongoose.Document {
  initialData: InitialData;
  weather_source: "machine" | "api" | "knn" | "calculations";
}

const SimulationInputSchema = new mongoose.Schema(
  {
    initialData: {
      type: initialDataSchema,
      required: true,
    },
    weather_source: {
      type: String,
      enum: ["machine", "api", "knn", "calculations"],
      required: true,
      default: "machine",
    },
  },
  defaultSchemaOptions,
);

export const SimulationInput = mongoose.model<SimulationInputDocument>(
  "SimulationInput",
  SimulationInputSchema,
);
