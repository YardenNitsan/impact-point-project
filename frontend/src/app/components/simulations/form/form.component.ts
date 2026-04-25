import { HttpClient } from '@angular/common/http';
import { Component, EventEmitter, Output } from '@angular/core';
import {
  FormGroup,
  FormControl,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { environment } from '../../../../../environment';
import { CommonModule } from '@angular/common';
import { finalize } from 'rxjs/operators';
import { Router } from '@angular/router';
import { SharedService } from '../../services/shared.service';
import { Coordinate } from '../../models/coordinate.model';
import { SimulationHistoryService } from '../history/history-services/simulationHistory.service';
import { SimulationTokenService } from '../../services/simulation-token.service';
import { SimulationRegistryService } from '../../services/simulation-registry.service';

type CreateSimulationResponse = {
  success: true;
  simulation: {
    id: string;
    createdAt: string;
    durationSeconds: number;
    weather_source: 'machine' | 'api' | 'calculations';
  };
  accessToken: string;
};

@Component({
  selector: 'app-form',
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './form.component.html',
  styleUrl: './form.component.css',
})
export class FormComponent {
  isLoadingOverlay = false;
  showSuccessModal = false;
  loading = false;
  errorMsg = '';
  simulationId?: string;

  constructor(
    private http: HttpClient,
    private shared: SharedService,
    private router: Router,
    private simulationService: SimulationHistoryService,
    private tokenService: SimulationTokenService,
    private registryService: SimulationRegistryService,
  ) {}

  trajectoryForm = new FormGroup({
    mass: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(800),
    ]),
    speed: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(514),
    ]),
    elevation: new FormControl('', [
      Validators.required,
      Validators.min(-35),
      Validators.max(45),
    ]),
    azimuth: new FormControl('', [
      Validators.required,
      Validators.min(0),
      Validators.max(359),
    ]),
    alt: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(19000),
    ]),
    lat: new FormControl('', [
      Validators.required,
      Validators.min(-90),
      Validators.max(90),
    ]),
    lon: new FormControl('', [
      Validators.required,
      Validators.min(-180),
      Validators.max(180),
    ]),
    weather_source: new FormControl<'machine' | 'api' | 'calculations'>(
      'machine',
      [Validators.required],
    ),
  });

  @Output() isSubmitted = new EventEmitter<boolean>();

  submit() {
    if (this.trajectoryForm.invalid) {
      this.trajectoryForm.markAllAsTouched();
      return;
    }

    if (this.loading) return;

    this.loading = true;
    this.isLoadingOverlay = true;
    this.errorMsg = '';

    const { mass, speed, elevation, azimuth, lat, lon, alt, weather_source } =
      this.trajectoryForm.value;

    const payload = {
      mass: Number(mass),
      initialSpeed: Number(speed),
      elevation: Number(elevation),
      azimuth: Number(azimuth),
      lat: Number(lat),
      lon: Number(lon),
      alt: Number(alt),
      weather_source: weather_source ?? 'machine',
    };

    this.http
      .post<CreateSimulationResponse>(
        environment.SIMULATION_REQUEST_URL,
        payload,
      )
      .pipe(
        finalize(() => {
          this.loading = false;
          this.isLoadingOverlay = false;
        }),
      )
      .subscribe({
        next: (body) => {
          this.simulationId = body?.simulation?.id;

          if (body?.simulation?.id && body?.accessToken) {
            this.tokenService.saveToken(body.simulation.id, body.accessToken);
            this.registryService.saveSimulation(body.simulation);
          }

          this.showSuccessModal = true;
          this.isSubmitted.emit(true);
        },
        error: (err) => {
          console.error('Simulation request failed:', err);

          this.errorMsg =
            err?.error?.error?.message ||
            err?.error?.message ||
            `Request failed with status ${err?.status ?? 'unknown'}`;
        },
      });
  }

  watchSimulation() {
    if (!this.simulationId) return;

    this.simulationService.watchSimulation(this.simulationId).subscribe({
      next: (coords: Coordinate[]) => {
        this.shared.setData(coords);
        this.router.navigateByUrl('/');
      },
      error: (err) =>
        alert(err?.message || err?.error?.error?.message || 'Load failed'),
    });
  }

  closeSuccessModal() {
    this.showSuccessModal = false;
    this.trajectoryForm.reset();
    this.trajectoryForm.markAsPristine();
    this.trajectoryForm.markAsUntouched();
  }
}
