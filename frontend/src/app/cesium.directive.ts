import { Directive, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { Viewer, Ion, createWorldTerrainAsync, createOsmBuildingsAsync } from 'cesium';
import { CesiumManager } from '../modules/cesium-manager-module';
import { environment } from '../../environment';

@Directive({ selector: '[appCesium]' })
export class CesiumDirective implements AfterViewInit, OnDestroy {
  private viewer?: Viewer;
  private ro?: ResizeObserver;

  constructor(private el: ElementRef<HTMLElement>, private cesiumManager: CesiumManager) {}

  async ngAfterViewInit(): Promise<void> {
    Ion.defaultAccessToken = environment.cesium_public_token;

    this.viewer = new Viewer(this.el.nativeElement, {
      timeline: true,
      animation: true,
      fullscreenButton: true,
      homeButton: true,
      geocoder: true,
      sceneModePicker: true,
      navigationHelpButton: true
    });

    this.viewer.terrainProvider = await createWorldTerrainAsync();
    this.viewer.scene.primitives.add(await createOsmBuildingsAsync());
    this.viewer.scene.globe.depthTestAgainstTerrain = true;

    this.cesiumManager.setViewer(this.viewer);

    // הכי חשוב: להתאים לכל שינוי גודל אמיתי של הקונטיינר
    this.ro = new ResizeObserver(() => {
      if (!this.viewer) return;
      this.viewer.resize();                 // ייעודי לזה :contentReference[oaicite:7]{index=7}
      this.viewer.scene.requestRender?.();
    });

    this.ro.observe(this.el.nativeElement);
  }

  ngOnDestroy(): void {
    this.ro?.disconnect();
    this.ro = undefined;

    this.viewer?.destroy();
    this.viewer = undefined;
  }
}
