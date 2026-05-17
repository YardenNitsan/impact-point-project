import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { environment } from '../../../../../../environment';
import { Coordinate } from '../../../models/coordinate.model';
import { SimulationTokenService } from '../../../services/simulation-token.service';

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
  weather_source: 'machine' | 'api' | 'knn' | 'calculations';
}

@Injectable({ providedIn: 'root' })
export class SimulationDetailsService {
  constructor(
    private http: HttpClient,
    private tokenService: SimulationTokenService,
  ) {}

  getDetails(id: string): Observable<SimulationDetails> {
    const token = this.tokenService.getToken(id);

    if (!token) {
      return throwError(
        () =>
          new Error(
            'No access token stored for this simulation in this browser.',
          ),
      );
    }

    const headers = new HttpHeaders({
      'x-simulation-token': token,
    });

    return this.http.get<SimulationDetails>(
      `${environment.SIMULATION_REQUEST_URL}/${id}/details`,
      { headers },
    );
  }
}
