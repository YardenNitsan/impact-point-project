import { Injectable } from '@angular/core';
import { Viewer } from 'cesium';
import { SharedService } from '../../services/shared.service';
import { Coordinate } from '../../models/coordinate.model';
import { filter, Subscription } from 'rxjs';
import {
  drawTrajectoryLOD,
  updateTrajectoryLOD,
  TrajectoryLODHandles,
} from './cesium-draw-module';

@Injectable({ providedIn: 'root' })
export class CesiumManager {
  private viewer?: Viewer;
  private dataSubscription?: Subscription;

  private trajectoryRenderState: TrajectoryLODHandles = { rawPoints: [] };
  private removeCameraListener?: () => void;

  constructor(private shared: SharedService) {}

  async initializeViewer(viewer: Viewer) {
    this.viewer = viewer;

    this.dataSubscription = this.shared.data$
      .pipe(
        filter(
          (coords): coords is Coordinate[] =>
            Array.isArray(coords) && coords.length > 0,
        ),
      )
      .subscribe((coords) => {
        this.drawTrajectoryAsync(coords);
      });
  }

  private async drawTrajectoryAsync(coords: Coordinate[]) {
    if (!this.viewer) return;

    const animationDurationSeconds = this.shared.lastSimulationDuration;

    this.trajectoryRenderState = await drawTrajectoryLOD(
      this.viewer,
      coords,
      this.trajectoryRenderState,
      animationDurationSeconds,
    );

    this.attachCameraLODListener();
  }

  private attachCameraLODListener() {
    if (!this.viewer || this.removeCameraListener) return;

    const viewer = this.viewer;

    const cameraMoveEndHandler = () => {
      if (!this.viewer) return;
      if (!this.trajectoryRenderState.rawPoints.length) return;

      updateTrajectoryLOD(this.viewer, this.trajectoryRenderState);
    };

    viewer.camera.moveEnd.addEventListener(cameraMoveEndHandler);

    this.removeCameraListener = () => {
      viewer.camera.moveEnd.removeEventListener(cameraMoveEndHandler);
      this.removeCameraListener = undefined;
    };
  }

  disposeViewer() {
    if (this.dataSubscription) {
      this.dataSubscription.unsubscribe();
      this.dataSubscription = undefined;
    }

    if (this.removeCameraListener) this.removeCameraListener();

    if (this.viewer) {
      if (this.trajectoryRenderState.polylineEntity)
        this.viewer.entities.remove(this.trajectoryRenderState.polylineEntity);

      if (this.trajectoryRenderState.movingEntity)
        this.viewer.entities.remove(this.trajectoryRenderState.movingEntity);

      this.viewer.trackedEntity = undefined;
      this.viewer.scene.requestRender();
    }

    this.viewer = undefined;
    this.trajectoryRenderState = { rawPoints: [] };
  }
}
