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

const commonConfig = {
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req: Request) => req.path === "/health",
} as const;

export const globalRateLimiter = rateLimit({
  ...commonConfig,
  windowMs: 60_000,
  max: 1000,
  keyGenerator: () => "global",
  handler: buildLimitHandler(
    "GLOBAL_RATE_LIMIT",
    "Server is busy. Please try again shortly.",
  ),
});

export const perIpRateLimiter = rateLimit({
  ...commonConfig,
  windowMs: 60_000,
  max: 120,
  handler: buildLimitHandler(
    "IP_RATE_LIMIT",
    "Too many requests from this IP. Please slow down.",
  ),
});

export const simulationCreateLimiter = rateLimit({
  ...commonConfig,
  windowMs: 60_000,
  max: 20,
  handler: buildLimitHandler(
    "SIMULATION_RATE_LIMIT",
    "Too many simulation requests from this IP. Please try again later.",
  ),
});
