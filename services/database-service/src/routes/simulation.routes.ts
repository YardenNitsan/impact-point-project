import { Router } from "express";
import { deleteSimulation } from "../controllers/simulationDelete.controller";
import { createSimulation } from "../controllers/simulationInput.controller";
import { getSimulationResults } from "../controllers/simulationResult.controller";
import { getSimulationToWatch } from "../controllers/simulationToWatch.controller";
import { getSimulationDetails } from "../controllers/simulationDetails.controller";

const router = Router();


router.delete("/:id", deleteSimulation);
router.post("/", createSimulation);
router.get("/", getSimulationResults);
router.get("/:id", getSimulationToWatch);
router.get("/:id/details", getSimulationDetails);


export default router;