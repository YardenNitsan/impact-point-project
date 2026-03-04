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
  ) {}

  trajectoryForm = new FormGroup({
    mass: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(5000),
    ]), // kg
    speed: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(1200),
    ]), // m/s
    elevation: new FormControl('', [
      Validators.required,
      Validators.min(-35),
      Validators.max(90),
    ]), // degrees
    azimuth: new FormControl('', [
      Validators.required,
      Validators.min(0),
      Validators.max(360),
    ]), // degrees
    alt: new FormControl('', [
      Validators.required,
      Validators.min(0),
      Validators.max(20000),
    ]),
    lat: new FormControl('', [Validators.required]),
    lon: new FormControl('', [Validators.required]),
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

    const { mass, speed, elevation, azimuth, lat, lon, alt } =
      this.trajectoryForm.value;
    const payload = {
      mass: Number(mass),
      initialSpeed: Number(speed),
      elevation: Number(elevation),
      azimuth: Number(azimuth),
      lat: Number(lat),
      lon: Number(lon),
      alt: Number(alt),
    };

    this.http
      .post(environment.SIMULATION_REQUEST_URL, payload, {
        observe: 'response',
      })
      .pipe(
        finalize(() => {
          this.loading = false;
          this.isLoadingOverlay = false;
        }),
      )
      .subscribe({
        next: (response) => {
          console.log('Server Response', response);
          this.simulationId = (response.body as any)?.resultId;

          this.showSuccessModal = true;

          this.isSubmitted.emit(true);
        },
        error: () => {
          this.errorMsg = 'Failed to send simulation. Please try again.';
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
      error: () => alert('load failed'),
    });
  }
  closeSuccessModal() {
    this.showSuccessModal = false;
    this.trajectoryForm.reset();
    this.trajectoryForm.markAsPristine();
    this.trajectoryForm.markAsUntouched();
  }
}
