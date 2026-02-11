import { Directive, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';

import {
  Ion,
  Viewer,
  createWorldTerrainAsync,
  createOsmBuildingsAsync,
  Cesium3DTileset,
} from 'cesium';

import { CesiumManager } from './modules/cesium-manager-module';
import { environment } from '../../../../environment';

/* =========================
   Buildings tuning
   ========================= */

// minimum distance for appearing buildings
const BUILDINGS_ENABLE_HEIGHT = 8000;

// maximum distance for appearing buildings
const BUILDINGS_DISABLE_HEIGHT = 15000;

// building's quality, lower is better performance
const BUILDINGS_MAX_SSE = 192;

@Directive({
  selector: '[appCesium]',
})
export class CesiumDirective implements AfterViewInit, OnDestroy {
  private viewer?: Viewer;
  private resizeObserver?: ResizeObserver;
  private buildings?: Cesium3DTileset;
  private cameraMoveEndHandler?: () => void;

  constructor(
    private el: ElementRef<HTMLElement>,
    private manager: CesiumManager,
  ) {}

  async ngAfterViewInit() {
    Ion.defaultAccessToken = environment.cesium_public_token;

    this.viewer = new Viewer(this.el.nativeElement, {
      timeline: true,
      animation: true,
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
    });

    this.viewer.terrainProvider = await createWorldTerrainAsync();

    this.buildings = await createOsmBuildingsAsync();
    this.buildings.maximumScreenSpaceError = BUILDINGS_MAX_SSE;

    this.buildings.show = false;
    this.viewer.scene.primitives.add(this.buildings);

    this.cameraMoveEndHandler = () => {
      if (!this.viewer || !this.buildings) return;

      const cameraHeightMeters = this.viewer.camera.positionCartographic.height;

      if (cameraHeightMeters < BUILDINGS_ENABLE_HEIGHT) {
        this.buildings.show = true;
      }

      if (cameraHeightMeters > BUILDINGS_DISABLE_HEIGHT) {
        this.buildings.show = false;
      }

      this.viewer.scene.requestRender();
    };

    this.viewer.camera.moveEnd.addEventListener(this.cameraMoveEndHandler);

    this.manager.initializeViewer(this.viewer);

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.viewer || this.viewer.isDestroyed()) return;

      this.viewer.resize();
      this.viewer.scene.requestRender();
    });

    this.resizeObserver.observe(this.el.nativeElement);

    this.viewer.scene.requestRender();
  }

  ngOnDestroy() {
    this.resizeObserver?.disconnect();
    this.resizeObserver = undefined;

    this.manager.disposeViewer();

    if (this.cameraMoveEndHandler && this.viewer) {
      this.viewer.camera.moveEnd.removeEventListener(this.cameraMoveEndHandler);
    }

    this.cameraMoveEndHandler = undefined;

    this.viewer?.destroy();
    this.viewer = undefined;
  }
}
