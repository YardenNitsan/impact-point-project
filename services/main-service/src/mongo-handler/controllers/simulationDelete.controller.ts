import { Request, Response } from "express";
import mongoose from "mongoose";

import { SimulationResult } from "../models/simulationResult.model";
import { SimulationInput } from "../models/simulationInput.model";

export const deleteSimulation = async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({
        message: "Invalid id"
      });
    }

    const result = await SimulationResult.findById(id);

    if (!result) {
      return res.status(404).json({
        message: "Simulation not found"
      });
    }

    await SimulationResult.findByIdAndDelete(id);

    if (result.simulationInputId) {
      await SimulationInput.findByIdAndDelete(
        result.simulationInputId
      );
    }

    res.json({
      message: "Simulation deleted successfully"
    });

  } catch (error: any) {
    console.error("Delete error:", error);

    res.status(500).json({
      message: "Delete failed",
      error: error.message
    });
  }
};
