import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CesiumDirective } from './cesium.directive';
import { FormComponent } from './components/form/form.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    CesiumDirective,
    FormComponent
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'frontend';
}
