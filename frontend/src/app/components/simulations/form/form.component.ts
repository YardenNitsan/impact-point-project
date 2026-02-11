import { HttpClient } from '@angular/common/http';
import { Component, EventEmitter, Output } from '@angular/core';
import { FormGroup, FormControl, ReactiveFormsModule } from '@angular/forms';
import { environment } from '../../../../../environment';
import { Validators } from '@angular/forms';

@Component({
  selector: 'app-form',
  imports: [ReactiveFormsModule],
  templateUrl: './form.component.html',
  styleUrl: './form.component.css',
})
export class FormComponent {
  private loading = false;

  constructor(private http: HttpClient) {}

  trajectoryForm = new FormGroup({
    mass: new FormControl('', [Validators.required]), // kg
    speed: new FormControl('', [Validators.required]), // m/s
    elevation: new FormControl('', [Validators.required]), // degrees
    azimuth: new FormControl('', [Validators.required]), // degrees
    lat: new FormControl('', [Validators.required]),
    lon: new FormControl('', [Validators.required]),
    alt: new FormControl('', [Validators.required]),
  });

  isOpen: boolean = false;
  toggle() {
    this.isOpen = !this.isOpen;
  }

  @Output() isSubmitted = new EventEmitter<boolean>();

  submit() {
    if (this.trajectoryForm.invalid) return;
    if (this.loading) return;
    this.loading = true;

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

    console.log('Payload:', payload);

    this.http
      .post(environment.SIMULATION_REQUEST_URL, payload, {
        observe: 'response',
      })
      .subscribe({
        next: (response) => {
          console.log('Server Response', response);
          this.isSubmitted.emit(true);
          this.loading = false;
        },
        error: (err) => {
          console.log('Error:', err);
          this.loading = false;
        },
      });
  }
}
