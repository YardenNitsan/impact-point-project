import { Request, Response, NextFunction } from "express";
import { ZodError } from "zod";

export function errorHandler(
  err: any,
  _req: Request,
  res: Response,
  _next: NextFunction,
) {
  if (err instanceof ZodError) {
    return res.status(400).json({
      success: false,
      error: {
        code: "VALIDATION_ERROR",
        message: "Invalid request input",
        details: err.flatten(),
      },
    });
  }

  const status = Number.isInteger(err?.status) ? err.status : 500;
  const code =
    typeof err?.code === "string" ? err.code : "INTERNAL_SERVER_ERROR";
  const publicMessage =
    status >= 500 ? "Internal server error" : err?.message || "Request failed";

  console.error("ERROR:", {
    code,
    status,
    message: err?.message,
    stack: err?.stack,
  });

  return res.status(status).json({
    success: false,
    error: {
      code,
      message: publicMessage,
    },
  });
}
