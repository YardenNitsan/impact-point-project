import { Request, Response } from "express";
import axios from "axios";

import { SimulationInput } from "../models/simulationInput.model";
import { SimulationResult } from "../models/simulationResult.model";
import { environmentService } from "../../environment";

export const createSimulation = async (req: Request, res: Response) => {
  const normalizedInput = {
    ...req.body,
    weather_source: req.body.weather_source ?? "machine",
  };

  const physicsPayload = {
    ...normalizedInput,
    return_trajectory: true,
    sample_dx_m: 12,
  };

  try {
    const pythonResponse = await axios.post(
      environmentService.PYTHON_SERVICE_URI,
      physicsPayload,
    );

    if (!pythonResponse.data || !pythonResponse.data.trajectory) {
      return res.status(400).json({
        error: "Python could not process the input data",
        details: "Invalid response from Python service",
      });
    }

    const inputDoc = new SimulationInput({
      initialData: normalizedInput,
      weather_source: normalizedInput.weather_source,
    });

    const savedInput = await inputDoc.save();

    const result = pythonResponse.data;

    const resultDoc = new SimulationResult({
      simulationInputId: savedInput._id,
      coordinates: result.trajectory,
      durationSeconds: Math.round(result.physical_time * 100) / 100,
      weather_source: normalizedInput.weather_source,
    });

    const savedResult = await resultDoc.save();

    res.status(201).json({
      inputId: savedInput._id,
      resultId: savedResult._id,
      algorithm: result,
    });
  } catch (error: any) {
    console.error("Simulation error:", error.message);

    res.status(500).json({
      error: "Failed to create simulation",
      details: error.message,
    });
  }
};
