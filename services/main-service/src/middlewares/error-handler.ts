import { error } from "console";
import { Request, Response, NextFunction } from "express";

export function errorHandler(
    err: any, 
    req: Request,
    res: Response,
    next: NextFunction
) {
    console.log("Error: ", err);

    const status = err.status || 500;
    
    res.status(status).json({
        success: false,
        error:{
            code: err.code || "Internal server error",  
            message: err.message || "Something went wrong"
        }
    });
}