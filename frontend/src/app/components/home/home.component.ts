import { Component } from '@angular/core';
import { CesiumDirective } from '../../cesium.directive';

@Component({
  selector: 'app-home',
  imports: [
    CesiumDirective,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {
  ngAfterViewInit() {
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 300);
  }

}
