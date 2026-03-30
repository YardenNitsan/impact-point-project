import { NextFunction, Request, Response } from "express";
import { z } from "zod";

const boundedNumber = (name: string, min: number, max: number) =>
  z
    .number({
      error: (issue) =>
        issue.input === undefined
          ? `${name} is required`
          : `${name} must be a number`,
    })
    .min(min, `${name} must be >= ${min}`)
    .max(max, `${name} must be <= ${max}`);

export const createSimulationBodySchema = z
  .object({
    lat: boundedNumber("lat", -90, 90),
    lon: boundedNumber("lon", -180, 180),
    alt: boundedNumber("alt", 0, 20000),
    azimuth: boundedNumber("azimuth", 0, 360),
    elevation: boundedNumber("elevation", -35, 85),
    mass: boundedNumber("mass", 1, 5000),
    initialSpeed: boundedNumber("initialSpeed", 1, 1200),
    weather_source: z
      .enum(["machine", "api", "calculations"])
      .default("machine"),
  })
  .strict();

const objectIdParamSchema = z.object({
  id: z.string().regex(/^[a-f\d]{24}$/i, "id must be a valid Mongo ObjectId"),
});

export function validateCreateSimulation(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  const parsed = createSimulationBodySchema.safeParse(req.body);

  if (!parsed.success) {
    return res.status(400).json({
      success: false,
      error: {
        code: "VALIDATION_ERROR",
        message: "Invalid simulation input",
        details: parsed.error.flatten(),
      },
    });
  }

  req.body = parsed.data;
  return next();
}

export function validateObjectIdParam(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  const parsed = objectIdParamSchema.safeParse(req.params);

  if (!parsed.success) {
    return res.status(400).json({
      success: false,
      error: {
        code: "INVALID_ID",
        message: "The supplied id is not a valid Mongo ObjectId",
      },
    });
  }

  return next();
}
