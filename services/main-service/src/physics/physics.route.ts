import { Router, Request, Response, NextFunction } from "express";
import { environmentService } from "../environment";

const router = Router();

const PHYSICS_SIM_URL = environmentService.PHYSICS_SIM_URL;

router.post("/", async (req: Request, res: Response, next: NextFunction) => {
  try {
    const response = await fetch(PHYSICS_SIM_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req.body),
    });

    const rawText = await response.text();

    let data: unknown = null;
    try {
      data = rawText ? JSON.parse(rawText) : null;
    } catch {
      data = rawText;
    }

    if (!response.ok) {
      return res.status(response.status).json({
        success: false,
        error: {
          code: "PHYSICS_SERVICE_ERROR",
          message: "Physics simulation service returned an error",
          details: data,
        },
      });
    }

    return res.status(200).json({
      success: true,
      data,
    });
  } catch (err) {
    next({
      status: 502,
      code: "PHYSICS_SERVICE_UNREACHABLE",
      message: "Could not reach physics simulation service",
      details: err,
    });
  }
});

export default router;
