import { Request, Response } from "express";
import { SimulationInput } from "../models/simulationInput.model";
import { SimulationLocals } from "../../middlewares/require-simulation-access-token";

export const getSimulationDetails = async (
  _req: Request,
  res: Response<any, SimulationLocals>,
) => {
  try {
    const result = res.locals.simulation;

    if (!result) {
      return res.status(404).json({
        success: false,
        error: {
          code: "SIMULATION_NOT_FOUND",
          message: "Simulation not found",
        },
      });
    }

    const input = await SimulationInput.findById(result.simulationInputId)
      .select("initialData")
      .lean();

    if (!input) {
      return res.status(404).json({
        success: false,
        error: {
          code: "SIMULATION_INPUT_NOT_FOUND",
          message: "Simulation input not found",
        },
      });
    }

    return res.status(200).json({
      createdAt: result.createdAt,
      durationSeconds: result.durationSeconds,
      initialData: input.initialData,
      coordinates: result.coordinates,
      weather_source: result.weather_source,
    });
  } catch (err) {
    console.error("DETAILS ERROR:", err);
    return res.status(500).json({
      success: false,
      error: {
        code: "LOAD_SIMULATION_DETAILS_FAILED",
        message: "Failed to load simulation details",
      },
    });
  }
};
