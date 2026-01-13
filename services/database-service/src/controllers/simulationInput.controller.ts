import { Request, Response } from "express";
import { SimulationInput } from "../models/simulationInput.model";
import { CreateSimulationBody } from "../types/simulation.types";

export const createSimulation = async (
  req: Request<{}, {}, CreateSimulationBody>,
  res: Response
) => {
  try {
    const simulation = new SimulationInput({
      initialData: {
        alt: req.body.alt,
        azimuth: req.body.azimuth,
        elevation: req.body.elevation,
        lat: req.body.lat,
        lon: req.body.lon,
        mass: req.body.mass,
        initialSpeed: req.body.speed
      }
    });

    const saved = await simulation.save();
    res.status(201).json(saved);
  } catch (err) {
    res.status(400).json({ error: "Failed to save simulation" });
  }
};

export const getSimulations = async (
  _req: Request,
  res: Response
) => {
  try {
    const simulations = await SimulationInput.find().sort({ createdAt: -1 });
    res.json(simulations);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch simulations" });
  }
};
