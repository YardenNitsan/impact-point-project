import mongoose from "mongoose";

const SimulationResultSchema = new mongoose.Schema(
  {
    simulationInputId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      unique: true,
    },
    coordinates: [
      {
        lon: Number,
        lat: Number,
        alt: Number,
      },
    ],
    durationMinutes: Number,
  },
  { timestamps: true }
);

export const SimulationResult = mongoose.model(
  "SimulationResult",
  SimulationResultSchema
);
