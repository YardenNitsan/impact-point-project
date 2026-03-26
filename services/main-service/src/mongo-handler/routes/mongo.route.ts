import { Router } from "express";
import { createSimulation } from "../controllers/simulationInput.controller";
import { getSimulationResults } from "../controllers/simulationResult.controller";
import { getSimulationToWatch } from "../controllers/simulationToWatch.controller";
import { getSimulationDetails } from "../controllers/simulationDetails.controller";
import { deleteSimulation } from "../controllers/simulationDelete.controller";

const router = Router();

router.post("/", createSimulation);
router.get("/", getSimulationResults);

router.get("/:id/details", getSimulationDetails);
router.get("/:id", getSimulationToWatch);

router.delete("/:id", deleteSimulation);

export default router;
