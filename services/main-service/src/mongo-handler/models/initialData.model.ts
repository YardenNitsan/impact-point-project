import mongoose from "mongoose";

export interface InitialData {
  alt: number;
  azimuth: number;
  elevation: number;
  lat: number;
  lon: number;
  mass: number;
  initialSpeed: number;
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
  },
  { _id: false }
);
