import { Router } from "express";
import {
  createSimulation,
  getSimulations
} from "../controllers/simulationInput.controller";

const router = Router();

router.post("/", createSimulation);
router.get("/", getSimulations);

export default router;
