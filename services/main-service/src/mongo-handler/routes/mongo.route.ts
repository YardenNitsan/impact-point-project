import { Router } from "express";
import { createSimulation } from "../controllers/simulationInput.controller";
import { getSimulationToWatch } from "../controllers/simulationToWatch.controller";
import { getSimulationDetails } from "../controllers/simulationDetails.controller";
import { deleteSimulation } from "../controllers/simulationDelete.controller";
import {
  validateCreateSimulation,
  validateObjectIdParam,
} from "../../middlewares/validate-request";
import { simulationCreateLimiter } from "../../middlewares/rate-limit";
import { requireSimulationAccessToken } from "../../middlewares/require-simulation-access-token";

const router = Router();

router.post(
  "/",
  simulationCreateLimiter,
  validateCreateSimulation,
  createSimulation,
);

router.get(
  "/:id/details",
  validateObjectIdParam,
  requireSimulationAccessToken,
  getSimulationDetails,
);

router.get(
  "/:id",
  validateObjectIdParam,
  requireSimulationAccessToken,
  getSimulationToWatch,
);

router.delete(
  "/:id",
  validateObjectIdParam,
  requireSimulationAccessToken,
  deleteSimulation,
);

export default router;
