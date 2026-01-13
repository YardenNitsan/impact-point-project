import { Router } from "express";
import { deleteSimulation } from "../controllers/simulationDelete.controller";
import { createSimulation } from "../controllers/simulationInput.controller";
import { getSimulationResults } from "../controllers/simulationResult.controller";

const router = Router();

router.delete("/:id", deleteSimulation);
router.post("/", createSimulation);
router.get("/", getSimulationResults);

export default router;