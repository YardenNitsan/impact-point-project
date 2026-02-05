import mongoose from "mongoose";

export const coordinateSchema = new mongoose.Schema(
  {
    lon: { type: Number, required: true },
    lat: { type: Number, required: true },
    alt: { type: Number, required: true },
  },
  { _id: false }
);
