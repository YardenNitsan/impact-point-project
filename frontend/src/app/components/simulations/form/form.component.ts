import { HttpClient } from '@angular/common/http';
import { Component, EventEmitter, Output } from '@angular/core';
import { FormGroup, FormControl, ReactiveFormsModule } from '@angular/forms';
import { environment } from '../../../../../environment';
import { Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { finalize } from 'rxjs/operators';

@Component({
  selector: 'app-form',
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './form.component.html',
  styleUrl: './form.component.css',
})
export class FormComponent {
  loading = false;
  success = false;
  errorMsg = '';

  constructor(private http: HttpClient) {}

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

  isOpen: boolean = false;
  toggle() {
    this.isOpen = !this.isOpen;
  }

  @Output() isSubmitted = new EventEmitter<boolean>();

  submit() {
    if (this.trajectoryForm.invalid) {
      this.trajectoryForm.markAllAsTouched();
      return;
    }
    if (this.loading) return;

    this.loading = true;
    this.success = false;
    this.errorMsg = '';

    const raw = this.trajectoryForm.value;
    const payload = {
      mass: Number(raw.mass),
      initialSpeed: Number(raw.speed),
      elevation: Number(raw.elevation),
      azimuth: Number(raw.azimuth),
      lat: Number(raw.lat),
      lon: Number(raw.lon),
      alt: Number(raw.alt),
    };

    this.http
      .post(environment.SIMULATION_REQUEST_URL, payload, {
        observe: 'response',
      })
      .pipe(finalize(() => (this.loading = false))) // יורד תמיד כשהבקשה מסתיימת
      .subscribe({
        next: () => {
          this.success = true;
          this.isSubmitted.emit(true);
          setTimeout(() => (this.success = false), 2500); // רק ההודעה נעלמת
        },
        error: () => {
          this.errorMsg = 'Failed to send simulation. Please try again.';
        },
      });
  }
}
