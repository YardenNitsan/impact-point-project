import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Observable } from "rxjs";

export interface CoordPoint {
  lat: number;
  lon: number;
  alt: number;
}

@Injectable({ providedIn: "root" })
export class CoordsApiService {

  constructor(private http: HttpClient) {}

  getCoords(): Observable<CoordPoint[]> {
    return this.http.get<CoordPoint[]>(
      "http://localhost:3000/api/coords"
    );
  }
}
