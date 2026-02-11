import { Component } from '@angular/core';
import { HistoryComponent } from './history/history.component';
import { FormComponent } from './form/form.component';
import { DetailsComponent } from './details/details.component';

type ViewMode = 'history' | 'form' | 'details';

@Component({
  selector: 'app-simulations',
  imports: [HistoryComponent, FormComponent, DetailsComponent],
  templateUrl: './simulations.component.html',
  styleUrl: './simulations.component.css'
})
export class SimulationsComponent {

   viewMode: ViewMode = 'history';
  total = 0;
  selectedSimulationId?: string;

  onCountChanged(value: number) {
    this.total = value;
  }

  toggleForm() {
    this.viewMode = this.viewMode === 'form' ? 'history' : 'form';
  }

  openDetails(id: string) {
    this.selectedSimulationId = id;
    this.viewMode = 'details';
  }

  backToHistory() {
    this.viewMode = 'history';
    this.selectedSimulationId = undefined;
  }
}
