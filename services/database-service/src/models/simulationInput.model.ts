import mongoose from "mongoose";

const SimulationInputSchema = new mongoose.Schema(
  {
    initialData: {
      alt: Number,
      azimuth: Number,
      elevation: Number,
      lat: Number,
      lon: Number,
      mass: Number,
      initialSpeed: Number
    }
  },
  {
    timestamps: {
      createdAt: true,
      updatedAt: false
    },
    versionKey: false
  }
);

export const SimulationInput = mongoose.model(
  "SimulationInput",
  SimulationInputSchema
);
