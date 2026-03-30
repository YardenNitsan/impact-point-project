import { Request, Response } from "express";
import { z } from "zod";
import { SimulationResult } from "../models/simulationResult.model";

const listSimulationQuerySchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(20).default(10),
});

export const getSimulationResults = async (req: Request, res: Response) => {
  const parsedQuery = listSimulationQuerySchema.safeParse(req.query);

  if (!parsedQuery.success) {
    return res.status(400).json({
      success: false,
      error: {
        code: "VALIDATION_ERROR",
        message: "Invalid list query",
        details: parsedQuery.error.flatten(),
      },
    });
  }

  const { page, limit } = parsedQuery.data;
  const skip = (page - 1) * limit;

  try {
    const [total, results] = await Promise.all([
      SimulationResult.countDocuments(),
      SimulationResult.find().sort({ createdAt: -1 }).skip(skip).limit(limit),
    ]);

    res.setHeader("X-Page", String(page));
    res.setHeader("X-Limit", String(limit));
    res.setHeader("X-Total-Count", String(total));
    res.setHeader("X-Total-Pages", String(Math.ceil(total / limit)));

    return res.status(200).json(
      results.map((r) => ({
        id: r._id.toString(),
        createdAt: r.createdAt,
        durationSeconds: r.durationSeconds,
        weather_source: r.weather_source,
      })),
    );
  } catch (err) {
    console.error(err);
    return res.status(500).json({
      success: false,
      error: {
        code: "FETCH_SIMULATIONS_FAILED",
        message: "Failed to fetch simulations",
      },
    });
  }
};
