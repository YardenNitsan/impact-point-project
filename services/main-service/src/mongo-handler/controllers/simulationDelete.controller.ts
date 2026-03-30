import { Request, Response } from "express";

import { SimulationResult } from "../models/simulationResult.model";
import { SimulationInput } from "../models/simulationInput.model";

export const deleteSimulation = async (req: Request, res: Response) => {
  try {
    const result = await SimulationResult.findById(req.params.id);

    if (!result) {
      return res.status(404).json({
        success: false,
        error: {
          code: "SIMULATION_NOT_FOUND",
          message: "Simulation not found",
        },
      });
    }

    await SimulationResult.findByIdAndDelete(req.params.id);

    if (result.simulationInputId) {
      await SimulationInput.findByIdAndDelete(result.simulationInputId);
    }

    return res.status(200).json({
      success: true,
      message: "Simulation deleted successfully",
    });
  } catch (error: any) {
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
