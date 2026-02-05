import mongoose from "mongoose";
import { defaultSchemaOptions } from "./SchemaOptions.model";
import {
  initialDataSchema,
  InitialData,
} from "./initialData.model";

export interface SimulationInputDocument extends mongoose.Document {
  initialData: InitialData;
}

const SimulationInputSchema = new mongoose.Schema(
  {
    initialData: {
      type: initialDataSchema,
      required: true,
    },
  },
  defaultSchemaOptions
);

export const SimulationInput = mongoose.model<SimulationInputDocument>(
  "SimulationInput",
  SimulationInputSchema
);
