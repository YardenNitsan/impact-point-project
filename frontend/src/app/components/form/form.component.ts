import { CommonModule } from '@angular/common';
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

  trajectoryForm = new FormGroup({
    mass: new FormControl(1),               // kg
    speed: new FormControl(100),            // m/s
    elevation: new FormControl(45),         // degrees
    azimuth: new FormControl(0),            // degrees
    lat: new FormControl(32.0853),
    lon: new FormControl(34.7818),
    alt: new FormControl(0)
  });

  isOpen:boolean = false;
  toggle(){
    this.isOpen = !this.isOpen;
  }
  close(){
    this.isOpen = false;
  }
  submit() {
    console.log('Form data:', this.trajectoryForm.value);
  }
}
