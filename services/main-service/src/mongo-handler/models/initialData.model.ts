import mongoose from "mongoose";

export interface InitialData {
  alt: number;
  azimuth: number;
  elevation: number;
  lat: number;
  lon: number;
  mass: number;
  initialSpeed: number;
  weather_source: "machine" | "api";
}

export const initialDataSchema = new mongoose.Schema<InitialData>(
  {
    alt: { type: Number, required: true },
    azimuth: { type: Number, required: true },
    elevation: { type: Number, required: true },
    lat: { type: Number, required: true },
    lon: { type: Number, required: true },
    mass: { type: Number, required: true },
    initialSpeed: { type: Number, required: true },
    weather_source: {
      type: String,
      enum: ["machine", "api"],
      required: true,
      default: "machine",
    },
  },
  { _id: false },
);
