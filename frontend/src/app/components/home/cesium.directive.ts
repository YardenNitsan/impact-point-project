import { Directive, ElementRef, AfterViewInit, OnDestroy } from "@angular/core";
import { Viewer, Ion, createWorldTerrainAsync, createOsmBuildingsAsync } from "cesium";
import { CesiumManager } from "./modules/cesium-manager-module";
import { environment } from "../../../../environment";

@Directive({ selector: "[appCesium]" })
export class CesiumDirective implements AfterViewInit, OnDestroy {

  private viewer?: Viewer;
  private ro?: ResizeObserver;

  constructor(
    //el is the element we are about to draw on it
    private el: ElementRef<HTMLElement>,
    private cesiumManager: CesiumManager
  ) {}

  //to make viewer in the DOM, ngOnInit a little fast sometimes. 
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

    //from here the world is ready 
    this.cesiumManager.setViewer(this.viewer);

    this.ro = new ResizeObserver(() => {
      if (!this.viewer || this.viewer.isDestroyed()) return;
      this.viewer.resize();
      this.viewer.scene.requestRender?.();
    });


    this.ro.observe(this.el.nativeElement);
  }

  ngOnDestroy(): void {
    this.ro?.disconnect();
    this.ro = undefined;

    this.cesiumManager.clearViewer();

    this.viewer?.destroy();
    this.viewer = undefined;
  }
}
