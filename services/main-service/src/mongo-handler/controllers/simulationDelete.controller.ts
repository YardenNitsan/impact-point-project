import { Request, Response } from "express";
import { SimulationInput } from "../models/simulationInput.model";
import { SimulationLocals } from "../../middlewares/require-simulation-access-token";

export const deleteSimulation = async (
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

    await result.deleteOne();

    if (result.simulationInputId) {
      await SimulationInput.findByIdAndDelete(result.simulationInputId);
    }

    return res.status(200).json({
      success: true,
      message: "Simulation deleted successfully",
    });
  } catch (error) {
    console.error("Delete error:", error);

    return res.status(500).json({
      success: false,
      error: {
        code: "DELETE_SIMULATION_FAILED",
        message: "Delete failed",
      },
    });
  }
};
