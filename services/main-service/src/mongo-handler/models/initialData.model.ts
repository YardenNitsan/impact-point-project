import mongoose from "mongoose";

export interface InitialData {
  alt: number;
  azimuth: number;
  elevation: number;
  lat: number;
  lon: number;
  mass: number;
  initialSpeed: number;
  weather_source: "machine" | "api" | "calculations";
}

export const initialDataSchema = new mongoose.Schema<InitialData>(
  {
    alt: { type: Number, required: true, min: 0, max: 20000 },
    azimuth: { type: Number, required: true, min: 0, max: 360 },
    elevation: { type: Number, required: true, min: -35, max: 85 },
    lat: { type: Number, required: true, min: -90, max: 90 },
    lon: { type: Number, required: true, min: -180, max: 180 },
    mass: { type: Number, required: true, min: 1, max: 5000 },
    initialSpeed: { type: Number, required: true, min: 1, max: 1200 },
    weather_source: {
      type: String,
      enum: ["machine", "api", "calculations"],
      required: true,
      default: "machine",
    },
  },
  { _id: false, strict: "throw" },
);
