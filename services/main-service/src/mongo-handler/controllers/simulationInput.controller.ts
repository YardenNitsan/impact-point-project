import { Request, Response } from "express";
import axios from "axios";

import { SimulationInput } from "../models/simulationInput.model";
import { SimulationResult } from "../models/simulationResult.model";
import { environmentService } from "../../environment";
import {
  generateSimulationAccessToken,
  SIMULATION_TOKEN_HEADER,
} from "../../utils/simulation-access-token";

const PHYSICS_RESPONSE_LIMIT_BYTES = 5 * 1024 * 1024;
const MAX_TRAJECTORY_POINTS = 30000;

function chooseSampleDx(input: { alt: number; initialSpeed: number }): number {
  const alt = Number(input.alt);
  const speed = Number(input.initialSpeed);

  if (alt >= 10000 || speed >= 800) return 5;
  if (alt >= 5000 || speed >= 500) return 4;
  return 3;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isTrajectoryPoint(
  value: unknown,
): value is { lon: number; lat: number; alt: number } {
  return (
    typeof value === "object" &&
    value !== null &&
    isFiniteNumber((value as { lon?: unknown }).lon) &&
    isFiniteNumber((value as { lat?: unknown }).lat) &&
    isFiniteNumber((value as { alt?: unknown }).alt)
  );
}

function isValidPhysicsResponse(value: unknown): value is {
  trajectory: { lon: number; lat: number; alt: number }[];
  physical_time: number;
} {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as { trajectory?: unknown[] }).trajectory) &&
    (value as { trajectory: unknown[] }).trajectory.length > 0 &&
    (value as { trajectory: unknown[] }).trajectory.length <=
      MAX_TRAJECTORY_POINTS &&
    (value as { trajectory: unknown[] }).trajectory.every(isTrajectoryPoint) &&
    isFiniteNumber((value as { physical_time?: unknown }).physical_time)
  );
}

export const createSimulation = async (req: Request, res: Response) => {
  const normalizedInput = req.body;

  const physicsPayload = {
    ...normalizedInput,
    return_trajectory: true,
    dx_sample_m: chooseSampleDx(normalizedInput),
  };

  try {
    const pythonResponse = await axios.post(
      environmentService.PYTHON_SERVICE_URI,
      physicsPayload,
      {
        timeout: 90_000,
        maxContentLength: PHYSICS_RESPONSE_LIMIT_BYTES,
        maxBodyLength: PHYSICS_RESPONSE_LIMIT_BYTES,
      },
    );

    if (!isValidPhysicsResponse(pythonResponse.data)) {
      return res.status(502).json({
        success: false,
        error: {
          code: "INVALID_PHYSICS_RESPONSE",
          message: "Physics service returned an invalid response",
        },
      });
    }

    const inputDoc = new SimulationInput({
      initialData: normalizedInput,
      weather_source: normalizedInput.weather_source,
    });

    const savedInput = await inputDoc.save();
    const { token: accessToken, tokenHash: accessTokenHash } =
      generateSimulationAccessToken();

    const resultDoc = new SimulationResult({
      simulationInputId: savedInput._id,
      coordinates: pythonResponse.data.trajectory,
      durationSeconds:
        Math.round(pythonResponse.data.physical_time * 100) / 100,
      weather_source: normalizedInput.weather_source,
      accessTokenHash,
    });

    const savedResult = await resultDoc.save();

    return res.status(201).json({
      success: true,
      inputId: savedInput._id,
      resultId: savedResult._id,
      accessToken,
      accessTokenHeader: SIMULATION_TOKEN_HEADER,
      algorithm: pythonResponse.data,
    });
  } catch (error: unknown) {
    if (axios.isAxiosError(error)) {
      console.error("AXIOS ERROR IN createSimulation:", {
        url: environmentService.PYTHON_SERVICE_URI,
        code: error.code,
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });

      if (error.code === "ECONNABORTED") {
        return res.status(504).json({
          success: false,
          error: {
            code: "PHYSICS_TIMEOUT",
            message: "Physics service timed out",
          },
        });
      }

      if (
        error.response &&
        error.response.status >= 400 &&
        error.response.status < 500
      ) {
        return res.status(400).json({
          success: false,
          error: {
            code: "PHYSICS_REJECTED_INPUT",
            message: "Physics service rejected the input",
            details: error.response.data,
          },
        });
      }

      if (error.response && error.response.status >= 500) {
        return res.status(502).json({
          success: false,
          error: {
            code: "PHYSICS_UPSTREAM_ERROR",
            message: "Physics service failed internally",
            details: error.response.data,
          },
        });
      }

      return res.status(502).json({
        success: false,
        error: {
          code: "PHYSICS_UNAVAILABLE",
          message: "Could not reach physics service",
        },
      });
    }

    console.error("Simulation error:", error);
    return res.status(500).json({
      success: false,
      error: {
        code: "CREATE_SIMULATION_FAILED",
        message: "Failed to create simulation",
      },
    });
  }
};
