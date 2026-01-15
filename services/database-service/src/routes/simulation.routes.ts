import { Router } from "express";
import { deleteSimulation } from "../controllers/simulationDelete.controller";
import { createSimulation } from "../controllers/simulationInput.controller";
import { getSimulationResults } from "../controllers/simulationResult.controller";
import { getSimulationToWatch } from "../controllers/simulationToWatch.controller";

const router = Router();

router.delete("/:id", deleteSimulation);
router.post("/", createSimulation);
router.get("/", getSimulationResults);
router.get("/:id", getSimulationToWatch);

export default router;