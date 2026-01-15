import { Request, Response } from "express"
import {SimulationResult} from "../models/simulationResult.model"

export const getSimulationToWatch = async (
    req: Request<{id: string}>, 
    res:Response
) =>{ 
    try{
        const {id} = req.params;
        const result = await SimulationResult.findById(id);

        if(!result)
            return res.status(404).json({message: 'object not found'});

        res.status(200).json(result.coordinates);
    } catch (err) {
        res.status(500).json({message: "failed to load simulation"});
    }
}