import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface SimulationHistoryItem {
  _id: string;
  createdAt: string;
  durationMinutes: number;
}

@Injectable({ providedIn: 'root' })
export class SimulationHistoryService {
  private API_URL = 'http://localhost:3000/api/simulation';

  constructor(private http: HttpClient) {}

  getSimulations(): Observable<SimulationHistoryItem[]> {
    return this.http.get<SimulationHistoryItem[]>(this.API_URL);
  }
  deleteSimulation(id: string): Observable<void> {
    return this.http.delete<void>(`${this.API_URL}/${id}`);
  }
}
