import mongoose from "mongoose";

// ===== Initial data (embedded) =====
export const initialDataSchema = new mongoose.Schema(
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
  { _id: false }
);

// ===== Single coordinate =====
export const coordinateSchema = new mongoose.Schema(
  {
    lon: Number,
    lat: Number,
    alt: Number
  },
  { _id: false }
);
