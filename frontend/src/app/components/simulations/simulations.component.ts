import { Component } from '@angular/core';
import { HistoryComponent } from './history/history.component';
import { FormComponent } from './form/form.component';

@Component({
  selector: 'app-simulations',
  imports: [HistoryComponent, FormComponent],
  templateUrl: './simulations.component.html',
  styleUrl: './simulations.component.css'
})
export class SimulationsComponent {
  formHistoryClicked: boolean = false;
  
  handleClick(){
    this.formHistoryClicked = !this.formHistoryClicked;
  }
}
