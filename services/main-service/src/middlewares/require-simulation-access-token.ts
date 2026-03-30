import { NextFunction, Request, Response } from "express";
import { SimulationResult } from "../mongo-handler/models/simulationResult.model";
import {
  SIMULATION_TOKEN_HEADER,
  simulationAccessTokenMatches,
} from "../utils/simulation-access-token";

export async function requireSimulationAccessToken(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  try {
    const providedToken = req.header(SIMULATION_TOKEN_HEADER)?.trim();

    if (!providedToken) {
      return res.status(401).json({
        success: false,
        error: {
          code: "SIMULATION_TOKEN_REQUIRED",
          message: `Missing ${SIMULATION_TOKEN_HEADER} header`,
        },
      });
    }

    const simulation = await SimulationResult.findById(req.params.id).select(
      "+accessTokenHash",
    );

    if (!simulation) {
      return res.status(404).json({
        success: false,
        error: {
          code: "SIMULATION_NOT_FOUND",
          message: "Simulation not found",
        },
      });
    }

    if (
      !simulation.accessTokenHash ||
      !simulationAccessTokenMatches(providedToken, simulation.accessTokenHash)
    ) {
      return res.status(403).json({
        success: false,
        error: {
          code: "SIMULATION_TOKEN_INVALID",
          message: "Invalid simulation access token",
        },
      });
    }

    return next();
  } catch (err) {
    return next(err);
  }
}
