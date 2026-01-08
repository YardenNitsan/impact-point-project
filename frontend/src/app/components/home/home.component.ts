import { Component } from '@angular/core';
import { PageNotFoundComponent } from '../page-not-found/page-not-found.component';
import { FormComponent } from './form/form.component';
import { CesiumDirective } from '../../cesium.directive';

@Component({
  selector: 'app-home',
  imports: [
    FormComponent,
    CesiumDirective
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {

}
