import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Coordinate } from '../../../models/coordinate.model';
import { environment } from '../../../../../../environment';
import { SimulationTokenService } from '../../../services/simulation-token.service';
import {
  SimulationRegistryService,
  StoredSimulation,
} from '../../../services/simulation-registry.service';

export interface SimulationHistoryItem extends StoredSimulation {
  formattedDuration?: string;
}

@Injectable({ providedIn: 'root' })
export class SimulationHistoryService {
  constructor(
    private http: HttpClient,
    private tokenService: SimulationTokenService,
    private registryService: SimulationRegistryService,
  ) {}

  getSimulations(): Observable<SimulationHistoryItem[]> {
    return this.registryService.simulations$;
  }

  private tokenHeaders(id: string): HttpHeaders {
    const token = this.tokenService.getToken(id);

    if (!token) {
      throw new Error(
        'No access token stored for this simulation in this browser.',
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
          this.registryService.removeSimulation(id);
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
