import { Injectable } from "@angular/core";
import { Viewer } from "cesium";
import { drawTrajectory } from "./cesium-draw-module";
import { SharedService, Coordinate } from "../../services/shared.service";
import { filter } from "rxjs";


//a class who manages between the viewer and coords
@Injectable({ providedIn: "root" })
export class CesiumManager {

  private viewer?: Viewer;

  constructor(private shared: SharedService) {}

  //check if there are coords in the memory and if there are, we will draw them. no race condition
  setViewer(viewer: Viewer) {
    this.viewer = viewer;
    console.log('Viewer is ready');

    this.shared.data$
      .pipe(filter((coords): coords is Coordinate[] => Array.isArray(coords)))
      .subscribe(coords => {
        drawTrajectory(this.viewer!, coords);
      });
  }

  clearViewer() {
    this.viewer = undefined;
  }
}
