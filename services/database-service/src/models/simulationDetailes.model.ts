import mongoose from "mongoose";
import { coordinateSchema, initialDataSchema } from "./subSchemas";
import { defaultSchemaOptions } from "./SchemaOptions.model";

const SimulationDetailesSchema = new mongoose.Schema(
    {
        initialData: initialDataSchema,
        coords: coordinateSchema,
        durationMinutes: {
            type: Number,
            required: true
        }
    },
    defaultSchemaOptions
)

export const SimulationDetailes = mongoose.model(
    "SimulationDetail",
    SimulationDetailesSchema
)