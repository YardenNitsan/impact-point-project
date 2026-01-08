import { Router, Request, Response } from "express";
import coords from "../controllers/coords-controller";

const router = Router();

router.get("/", (res: Response) => {
  res.json(coords);
});

export default router;
