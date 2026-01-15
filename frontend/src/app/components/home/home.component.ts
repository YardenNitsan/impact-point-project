import { Component } from '@angular/core';
import { CesiumDirective } from './cesium.directive';
import { Coordinate, SharedService } from '../services/shared.service';
import { Subject, takeUntil } from 'rxjs';
import { Viewer } from 'cesium';

@Component({
  selector: 'app-home',
  imports: [
    CesiumDirective,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {

}
