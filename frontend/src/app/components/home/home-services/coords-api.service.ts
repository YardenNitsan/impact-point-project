import { Injectable } from "@angular/core";
import {SharedService } from "../../services/shared.service";

@Injectable({ providedIn: 'root' })
export class CoordsApiService {
  constructor(private shared: SharedService) {}

  getCoords$() {
    return this.shared.data$;
  }
}
