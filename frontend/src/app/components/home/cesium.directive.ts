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

// building's quality, lower is better performance
const BUILDINGS_MAX_SSE = 256;

@Directive({
  selector: '[appCesium]',
})
export class CesiumDirective implements AfterViewInit, OnDestroy {
  private viewer?: Viewer;
  private resizeObserver?: ResizeObserver;
  private buildings?: Cesium3DTileset;

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

    this.buildings.show = true;
    this.viewer.scene.primitives.add(this.buildings);

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

    this.viewer?.destroy();
    this.viewer = undefined;
  }
}
