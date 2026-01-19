import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface InitialData {
  alt: number;
  azimuth: number;
  elevation: number;
  lat: number;
  lon: number;
  mass: number;
  initialSpeed: number;
}

export interface Coordinate {
  lon: number;
  lat: number;
  alt: number;
}

export interface SimulationDetails {
  createdAt: string;
  durationMinutes: number;
  initialData: InitialData;
  coordinates: Coordinate[];
}

@Injectable({ providedIn: 'root' })
export class SimulationDetailsService {
  private BASE_URL = 'http://localhost:3000/api/simulation';

  constructor(private http: HttpClient) {}

  getDetails(id: string): Observable<SimulationDetails> {
    return this.http.get<SimulationDetails>(`${this.BASE_URL}/${id}/details`);
  }
}
