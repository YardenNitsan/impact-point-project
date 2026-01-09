import { Directive, ElementRef, AfterViewInit } from '@angular/core';
import {
  Viewer,
  Ion,
  createWorldTerrainAsync,
  createOsmBuildingsAsync
} from 'cesium';
import { CesiumManager } from '../modules/cesium-manager-module';
import { environment } from '../../environment';

@Directive({
  selector: '[appCesium]'
})
export class CesiumDirective implements AfterViewInit {

  constructor(
    private el: ElementRef,
    private cesiumManager: CesiumManager
  ) {}

  async ngAfterViewInit(): Promise<void> {
    Ion.defaultAccessToken = environment.cesium_public_token;

    const viewer = new Viewer(this.el.nativeElement, {
      timeline: true,
      animation: true,
      fullscreenButton: true,
      homeButton: true,
      geocoder: true,
      sceneModePicker: true,
      navigationHelpButton: true
    });

    viewer.terrainProvider = await createWorldTerrainAsync();
    viewer.scene.primitives.add(await createOsmBuildingsAsync());
    viewer.scene.globe.depthTestAgainstTerrain = true;

    this.cesiumManager.setViewer(viewer);
  }
}
