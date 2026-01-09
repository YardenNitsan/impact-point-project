import { Component } from '@angular/core';
import { PageNotFoundComponent } from '../page-not-found/page-not-found.component';
import { FormComponent } from './form/form.component';
import { CesiumDirective } from '../../cesium.directive';
import { SideBarComponent } from "../side-bar/side-bar.component";

@Component({
  selector: 'app-home',
  imports: [
    FormComponent,
    CesiumDirective,
    SideBarComponent
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
