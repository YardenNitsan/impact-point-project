import { Request, Response, NextFunction } from "express";
import { ZodError } from "zod";

function normalizeStatus(value: unknown): number {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 400 &&
    value <= 599
    ? value
    : 500;
}

export function errorHandler(
  err: unknown,
  _req: Request,
  res: Response,
  next: NextFunction,
) {
  if (res.headersSent) {
    return next(err);
  }

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

  const maybeBodyParserError =
    typeof err === "object" && err !== null
      ? (err as { type?: unknown })
      : undefined;

  if (maybeBodyParserError?.type === "entity.too.large") {
    return res.status(413).json({
      success: false,
      error: {
        code: "PAYLOAD_TOO_LARGE",
        message: "Request body is too large",
      },
    });
  }

  const error =
    typeof err === "object" && err !== null
      ? (err as {
          status?: unknown;
          statusCode?: unknown;
          code?: unknown;
          message?: unknown;
          stack?: unknown;
        })
      : {};

  const status = normalizeStatus(error.status ?? error.statusCode);
  const code =
    typeof error.code === "string" ? error.code : "INTERNAL_SERVER_ERROR";
  const publicMessage =
    status >= 500
      ? "Internal server error"
      : typeof error.message === "string" && error.message.trim()
        ? error.message
        : "Request failed";

  const logPayload = {
    code,
    status,
    message: typeof error.message === "string" ? error.message : undefined,
    stack: typeof error.stack === "string" ? error.stack : undefined,
  };

  if (status >= 500) {
    console.error("ERROR:", logPayload);
  } else {
    console.warn("REQUEST_ERROR:", logPayload);
  }

  return res.status(status).json({
    success: false,
    error: {
      code,
      message: publicMessage,
    },
  });
}
