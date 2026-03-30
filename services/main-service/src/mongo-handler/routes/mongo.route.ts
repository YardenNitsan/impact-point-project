import { Router } from "express";
import { createSimulation } from "../controllers/simulationInput.controller";
import { getSimulationResults } from "../controllers/simulationResult.controller";
import { getSimulationToWatch } from "../controllers/simulationToWatch.controller";
import { getSimulationDetails } from "../controllers/simulationDetails.controller";
import { deleteSimulation } from "../controllers/simulationDelete.controller";
import {
  validateCreateSimulation,
  validateObjectIdParam,
} from "../../middlewares/validate-request";
import { simulationCreateLimiter } from "../../middlewares/rate-limit";

const router = Router();

router.post(
  "/",
  simulationCreateLimiter,
  validateCreateSimulation,
  createSimulation,
);
router.get("/", getSimulationResults);

router.get("/:id/details", validateObjectIdParam, getSimulationDetails);
router.get("/:id", validateObjectIdParam, getSimulationToWatch);

router.delete("/:id", validateObjectIdParam, deleteSimulation);

export default router;
