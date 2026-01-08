import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormGroup, FormControl, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-form',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    CommonModule
  ],
  templateUrl: './form.component.html',
  styleUrl: './form.component.css'
})
export class FormComponent {

  constructor(private http: HttpClient){

  }
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

  submit() {
    if(this.trajectoryForm.invalid){
      return;
    }
    const payload = this.trajectoryForm.value;
    console.log('Payload: ', payload);
    this.http.post('http://localhost:3000/api/trajectory',
      payload
    ).subscribe({
      next: (response) => {
        console.log('Server Resonse', response);
      },
      error: (err) => {
        console.log('Error: ', err);
      }
    })
  }
}
