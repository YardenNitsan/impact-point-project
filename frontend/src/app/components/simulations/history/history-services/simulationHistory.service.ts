import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Coordinate } from '../../../models/coordinate.model';
import { environment } from '../../../../../../environment';

export interface SimulationHistoryItem {
  id: string;
  createdAt: string;
  durationSeconds: number;
  weather_source: 'machine' | 'api' | 'calculations';
  formattedDuration?: string;
}

@Injectable({ providedIn: 'root' })
export class SimulationHistoryService {
  constructor(private http: HttpClient) {}

  getSimulations(): Observable<SimulationHistoryItem[]> {
    return this.http.get<SimulationHistoryItem[]>(
      environment.SIMULATION_REQUEST_URL,
    );
  }
  deleteSimulation(id: string): Observable<void> {
    return this.http.delete<void>(
      `${environment.SIMULATION_REQUEST_URL}/${id}`,
    );
  }
  watchSimulation(id: string): Observable<Coordinate[]> {
    return this.http.get<Coordinate[]>(
      `${environment.SIMULATION_REQUEST_URL}/${id}`,
    );
  }
}
