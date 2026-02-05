import {
  Directive,
  ElementRef,
  AfterViewInit,
  OnDestroy
} from "@angular/core";

import {
  Ion,
  Viewer,
  createWorldTerrainAsync,
  createOsmBuildingsAsync,
  Cesium3DTileset
} from "cesium";

import { CesiumManager } from "./modules/cesium-manager-module";
import { environment } from "../../../../environment";

/* =========================
   Buildings tuning
   ========================= */

// בניינים מופיעים רק כשקרובים
const BUILDINGS_ENABLE_HEIGHT = 8000;

// מעל זה → נכבים
const BUILDINGS_DISABLE_HEIGHT = 15000;

// איכות הבניינים (גבוה = פחות עומס)
const BUILDINGS_MAX_SSE = 192;

@Directive({
  selector: "[appCesium]",
})
export class CesiumDirective
  implements AfterViewInit, OnDestroy
{
  private viewer?: Viewer;
  private resizeObserver?: ResizeObserver;
  private buildings?: Cesium3DTileset;

  constructor(
    private el: ElementRef<HTMLElement>,
    private manager: CesiumManager
  ) {}

  async ngAfterViewInit() {
    Ion.defaultAccessToken = environment.cesium_public_token;

    this.viewer = new Viewer(this.el.nativeElement, {
      timeline: true,
      animation: true,
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
    });

    this.viewer.terrainProvider =
      await createWorldTerrainAsync();

    // 🔥 טוען בניינים פעם אחת בלבד
    this.buildings = await createOsmBuildingsAsync();
    this.buildings.maximumScreenSpaceError =
      BUILDINGS_MAX_SSE;

    this.buildings.show = false; // מתחיל כבוי
    this.viewer.scene.primitives.add(this.buildings);

    // 🔥 מערכת הפעלה חכמה לבניינים
    this.viewer.camera.moveEnd.addEventListener(() => {
      if (!this.viewer || !this.buildings) return;

      const h =
        this.viewer.camera.positionCartographic.height;

      if (h < BUILDINGS_ENABLE_HEIGHT) {
        this.buildings.show = true;
      }

      if (h > BUILDINGS_DISABLE_HEIGHT) {
        this.buildings.show = false;
      }

      this.viewer.scene.requestRender();
    });

    // manager
    this.manager.setViewer(this.viewer);

    // resize handling
    this.resizeObserver = new ResizeObserver(() => {
      if (!this.viewer || this.viewer.isDestroyed())
        return;

      this.viewer.resize();
      this.viewer.scene.requestRender();
    });

    this.resizeObserver.observe(this.el.nativeElement);

    this.viewer.scene.requestRender();
  }

  ngOnDestroy() {
    this.resizeObserver?.disconnect();
    this.resizeObserver = undefined;

    this.manager.clearViewer();

    this.viewer?.destroy();
    this.viewer = undefined;
  }
}
