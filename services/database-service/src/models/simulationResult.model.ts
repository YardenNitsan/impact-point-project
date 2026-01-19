import mongoose from "mongoose";
import { coordinateSchema } from "./subSchemas";
import { defaultSchemaOptions } from "./SchemaOptions.model";

const SimulationResultSchema = new mongoose.Schema(
  {
    simulationInputId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "SimulationInput",
      required: true
    },

    coordinates: {
      type: [coordinateSchema],
      required: true,
      default: []
    },

    durationMinutes: {
      type: Number,
      required: true
    }
  },
  defaultSchemaOptions
);

export const SimulationResult = mongoose.model(
  "SimulationResult",
  SimulationResultSchema
);
