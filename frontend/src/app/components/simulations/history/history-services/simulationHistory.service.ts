import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Coordinate } from '../../../models/coordinate.model';
import { environment } from '../../../../../../environment';
import { SimulationTokenService } from '../../../services/simulation-token.service';

export interface SimulationHistoryItem {
  id: string;
  createdAt: string;
  durationSeconds: number;
  weather_source: 'machine' | 'api' | 'calculations';
  formattedDuration?: string;
}

@Injectable({ providedIn: 'root' })
export class SimulationHistoryService {
  constructor(
    private http: HttpClient,
    private tokenService: SimulationTokenService,
  ) {}

  getSimulations(): Observable<SimulationHistoryItem[]> {
    return this.http.get<SimulationHistoryItem[]>(
      environment.SIMULATION_REQUEST_URL,
    );
  }

  private tokenHeaders(id: string): HttpHeaders {
    const token = this.tokenService.getToken(id);

    if (!token) {
      throw new Error(
        'No access token stored for this simulation. Create it again after the new security fix.',
      );
    }

    return new HttpHeaders({
      'x-simulation-token': token,
    });
  }

  deleteSimulation(id: string): Observable<void> {
    let headers: HttpHeaders;

    try {
      headers = this.tokenHeaders(id);
    } catch (err: any) {
      return throwError(() => err);
    }

    return this.http
      .delete<void>(`${environment.SIMULATION_REQUEST_URL}/${id}`, { headers })
      .pipe(
        tap(() => {
          this.tokenService.removeToken(id);
        }),
      );
  }

  watchSimulation(id: string): Observable<Coordinate[]> {
    let headers: HttpHeaders;

    try {
      headers = this.tokenHeaders(id);
    } catch (err: any) {
      return throwError(() => err);
    }

    return this.http.get<Coordinate[]>(
      `${environment.SIMULATION_REQUEST_URL}/${id}`,
      { headers },
    );
  }
}
