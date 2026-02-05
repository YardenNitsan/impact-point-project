import { Injectable } from "@angular/core";
import { Viewer } from "cesium";
import { SharedService, Coordinate } from "../../services/shared.service";
import { filter } from "rxjs";
import {
  drawTrajectoryLOD,
  updateTrajectoryLOD,
  TrajectoryLODHandles,
} from "./cesium-draw-module";

@Injectable({ providedIn: "root" })
export class CesiumManager {
  private viewer?: Viewer;

  private handles: TrajectoryLODHandles = { rawPoints: [] };
  private removeCameraListener?: () => void;

  constructor(private shared: SharedService) {}

  async setViewer(viewer: Viewer) {
    this.viewer = viewer;

    this.shared.data$
      .pipe(
        filter(
          (coords): coords is Coordinate[] =>
            Array.isArray(coords) && coords.length > 0
        )
      )
      .subscribe((coords) => {
        this.drawAsync(coords);
      });
  }

  private async drawAsync(coords: Coordinate[]) {
    if (!this.viewer) return;

    const duration = this.shared.lastSimulationDuration;

    // Heavy only when NEW data arrives (but entities are reused)
    this.handles = await drawTrajectoryLOD(
      this.viewer,
      coords,
      this.handles,
      duration
    );

    this.attachLODListener();
  }

  private attachLODListener() {
    if (!this.viewer || this.removeCameraListener) return;

    const viewer = this.viewer;

    const handler = () => {
      if (!this.viewer) return;
      if (!this.handles.rawPoints.length) return;

      // ✅ ultra-light update only
      updateTrajectoryLOD(this.viewer, this.handles);
    };

    viewer.camera.moveEnd.addEventListener(handler);

    this.removeCameraListener = () => {
      viewer.camera.moveEnd.removeEventListener(handler);
      this.removeCameraListener = undefined;
    };
  }

  clearViewer() {
    if (this.removeCameraListener) this.removeCameraListener();

    if (this.viewer) {
      if (this.handles.polylineEntity)
        this.viewer.entities.remove(this.handles.polylineEntity);

      if (this.handles.movingEntity)
        this.viewer.entities.remove(this.handles.movingEntity);

      this.viewer.trackedEntity = undefined;
      this.viewer.scene.requestRender();
    }

    this.viewer = undefined;
    this.handles = { rawPoints: [] };
  }
}
