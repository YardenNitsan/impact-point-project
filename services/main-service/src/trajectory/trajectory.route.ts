import {Router} from "express";
import { NextFunction } from "express";

const routerTrajectory = Router();

routerTrajectory.post('/', (req, res, next) => {
    if(!req.body){
        return next({
            status: 400, 
            code: "Empty body",
            message: "Request body is required"
        });
    }

    console.log("Data received: ", req.body);

    res.json({
        status: 200,
        message: "Data received successfully"
    });
});

export default routerTrajectory;