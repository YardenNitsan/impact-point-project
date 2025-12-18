import { Injectable } from "@angular/core";
import { Viewer } from "cesium";
import { CoordsApiService } from "../services/coords-api.service";
import { drawTrajectory } from "./cesium-draw-module";

@Injectable({ providedIn: "root" })
export class CesiumManager {

  private viewer!: Viewer;
  
  constructor(private coordsApi: CoordsApiService) {}

  setViewer(viewer: Viewer) {
    this.viewer = viewer;

    this.loadAndDraw();
  }

  private loadAndDraw() {
    this.coordsApi.getCoords().subscribe(points => {
      console.log("Coords from server:", points);
      drawTrajectory(this.viewer, points);
    });
  }

  getViewer(): Viewer {
    return this.viewer;
  }
}
