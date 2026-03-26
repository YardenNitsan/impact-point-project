import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../../../../environment';
import { Coordinate } from '../../../models/coordinate.model';

export interface InitialData {
  alt: number;
  azimuth: number;
  elevation: number;
  lat: number;
  lon: number;
  mass: number;
  initialSpeed: number;
}

export interface SimulationDetails {
  createdAt: string;
  durationSeconds: number;
  initialData: InitialData;
  coordinates: Coordinate[];
}

@Injectable({ providedIn: 'root' })
export class SimulationDetailsService {
  constructor(private http: HttpClient) {}

  getDetails(id: string): Observable<SimulationDetails> {
    return this.http.get<SimulationDetails>(
      `${environment.SIMULATION_REQUEST_URL}/${id}/details`,
    );
  }
}
