import mongoose from "mongoose";
import { initialDataSchema } from "./subSchemas";
import { defaultSchemaOptions } from "./SchemaOptions.model";

const SimulationInputSchema = new mongoose.Schema(
  {
    initialData: initialDataSchema
  },
  defaultSchemaOptions
);

export const SimulationInput = mongoose.model(
  "SimulationInput",
  SimulationInputSchema
);
