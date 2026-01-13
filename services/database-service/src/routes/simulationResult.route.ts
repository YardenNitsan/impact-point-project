import { Router } from "express";
import { getSimulationResults } from "../controllers/simulationResult.controller";

const router = Router();

router.get("/", getSimulationResults);

export default router;
