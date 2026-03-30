import { Request, Response } from "express";
import { rateLimit } from "express-rate-limit";

const buildLimitHandler =
  (code: string, message: string) => (_req: Request, res: Response) => {
    res.status(429).json({
      success: false,
      error: {
        code,
        message,
      },
    });
  };

export const globalRateLimiter = rateLimit({
  windowMs: 60_000,
  max: 1000,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: () => "global",
  handler: buildLimitHandler(
    "GLOBAL_RATE_LIMIT",
    "Server is busy. Please try again shortly.",
  ),
});

export const perIpRateLimiter = rateLimit({
  windowMs: 60_000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
  handler: buildLimitHandler(
    "IP_RATE_LIMIT",
    "Too many requests from this IP. Please slow down.",
  ),
});

export const simulationCreateLimiter = rateLimit({
  windowMs: 60_000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  handler: buildLimitHandler(
    "SIMULATION_RATE_LIMIT",
    "Too many simulation requests from this IP. Please try again later.",
  ),
});
