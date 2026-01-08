import { Router } from "express";
import { coordinates } from "./coords.logic"; 

const routerCoords = Router();

routerCoords.get('/', (_req, res) =>{
    const coordsArray = coordinates();
    res.json(coordsArray);
});

export default routerCoords;