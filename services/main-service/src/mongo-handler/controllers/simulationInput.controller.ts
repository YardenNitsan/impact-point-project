// import { Request, Response } from "express";
// import { SimulationInput } from "../models/simulationInput.model";

// export const createSimulation = async (
//   req: Request,
//   res: Response
// ) => {
//   try {
//     const simulation = new SimulationInput({
//       initialData: req.body,
//     });

//     const saved = await simulation.save();
//     res.status(201).json(saved);
//   } catch {
//     res.status(400).json({ error: "Failed to save simulation" });
//   }
// };
// import { Request, Response } from "express";
// import axios from "axios";

// export const createSimulation = async (
//   req: Request,
//   res: Response
// ) => {
//   try {
//     // שולח את התנאים ההתחלתיים לשרת הפייתון
//     const pythonResponse = await axios.post(
//       "http://127.0.0.1:8000/simulate-impact",
//       req.body
//     );

//     // מחזיר לפרונט את תוצאת האלגוריתם
//     res.status(200).json(pythonResponse.data);

//   } catch (error: any) {
//     console.error("Python service error:", error.message);

//     res.status(500).json({
//       error: "Failed to run simulation",
//       details: error.message
//     });
//   }
// };
import { Request, Response } from "express";
import axios from "axios";

import {SimulationInput} from "../models/simulationInput.model";
import {SimulationResult} from "../models/simulationResult.model";

export const createSimulation = async (req: Request, res: Response) => {
  try {
    // 1. Save initial input to Mongo
    const inputDoc = new SimulationInput({
      initialData: req.body
    });

    const savedInput = await inputDoc.save();

    // 2. Call Python
    const pythonResponse = await axios.post(
      "http://127.0.0.1:8000/simulate-impact",
      req.body
    );

    const result = pythonResponse.data;

    // 3. Save result linked to input
    const resultDoc = new SimulationResult({
      simulationInputId: savedInput._id,
      coordinates: result.trajectory,
      durationSeconds: Math.round(result.physical_time * 100) / 100
    });

    const savedResult = await resultDoc.save();

    // 4. Return response
    res.status(201).json({
      inputId: savedInput._id,
      resultId: savedResult._id,
      algorithm: result
    });

  } catch (error: any) {
    console.error("Simulation error:", error.message);

    res.status(500).json({
      error: "Failed to create simulation",
      details: error.message
    });
  }
};
