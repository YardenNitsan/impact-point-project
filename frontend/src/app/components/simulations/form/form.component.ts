import { HttpClient } from '@angular/common/http';
import { Component, EventEmitter, Output } from '@angular/core';
import { FormGroup, FormControl, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-form',
  imports: [
    ReactiveFormsModule,
  ],
  templateUrl: './form.component.html',
  styleUrl: './form.component.css'
})
export class FormComponent {

  constructor(private http: HttpClient){}

  trajectoryForm = new FormGroup({
    mass: new FormControl(''),               // kg
    speed: new FormControl(''),            // m/s
    elevation: new FormControl(''),         // degrees
    azimuth: new FormControl(''),            // degrees
    lat: new FormControl(''),
    lon: new FormControl(''),
    alt: new FormControl('')
  });

  isOpen:boolean = false;
  toggle(){
    this.isOpen = !this.isOpen;
  }

  @Output() isSubmitted = new EventEmitter<boolean>();

  submit() {
    if (this.trajectoryForm.invalid) return;

    const raw = this.trajectoryForm.value;

    const payload = {
      mass: Number(raw.mass),
      initialSpeed: Number(raw.speed),
      elevation: Number(raw.elevation),
      azimuth: Number(raw.azimuth),
      lat: Number(raw.lat),
      lon: Number(raw.lon),
      alt: Number(raw.alt)
    };

    console.log('Payload:', payload);

    this.http.post(
      'http://localhost:3000/api/simulation',
      payload,
      { observe: 'response' }
    ).subscribe({
      next: (response) => {
        console.log('Server Response', response);
        this.isSubmitted.emit(true);
      },
      error: (err) => {
        console.log('Error:', err);
      }
    });
  }
}
