import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

export const getSimulationToWatch = async (req: Request, res: Response) => {
  try {
    const sim = await SimulationResult.findById(req.params.id).select(
      "coordinates",
    );

    if (!sim) {
      return res.status(404).json({
        success: false,
        error: {
          code: "SIMULATION_NOT_FOUND",
          message: "Simulation not found",
        },
      });
    }

    return res.status(200).json(sim.coordinates);
  } catch (err) {
    console.error(err);
    return res.status(500).json({
      success: false,
      error: {
        code: "LOAD_SIMULATION_FAILED",
        message: "Failed to load simulation",
      },
    });
  }
};
