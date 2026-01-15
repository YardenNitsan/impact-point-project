import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";
import { SimulationInput } from "../models/simulationInput.model";

export const deleteSimulation = async (
  req: Request<{ id: string }>,
  res: Response
) => {
  try {
    const { id } = req.params;
    console.log('id:',id);

    const result = await SimulationResult. findById(id);
    if(!result)
      return res.status(404).json({error: 'sim not found'});

    const deletedResult = await SimulationResult.findByIdAndDelete(id);
    if(!deletedResult)
      return res.status(404).json({ error: "Simulation not found" });

    const inputID = result.simulationInputId;    
    const deletedInput = await SimulationInput.findByIdAndDelete(inputID);
    if(!deletedInput)
      return res.status(404).json({ error: "Simulation not found" });

    res.status(200).json({ message: "Simulation deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete simulation" });
  }
};
