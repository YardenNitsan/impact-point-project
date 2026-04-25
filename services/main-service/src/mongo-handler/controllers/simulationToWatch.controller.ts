import { Request, Response } from "express";
import { SimulationLocals } from "../../middlewares/require-simulation-access-token";

export const getSimulationToWatch = async (
  _req: Request,
  res: Response<any, SimulationLocals>,
) => {
  try {
    const sim = res.locals.simulation;

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
