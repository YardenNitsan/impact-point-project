import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SimulationHistoryService, SimulationHistoryItem } from './services/simulationHistory.service';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent implements OnInit {

  simulations: SimulationHistoryItem[] = [];
  @Output() countSim = new EventEmitter<number>();

  cardsToNotScroll = 5;
  count = 0;
  shouldScroll = false;

  constructor(
    private simulationService: SimulationHistoryService
  ) {}

  ngOnInit(): void {
    this.simulationService.getSimulations().subscribe({
      next: (data) => {
        this.simulations = data;
        this.countSim.emit(data.length);
        this.shouldScroll = data.length > this.cardsToNotScroll;
      },
      error: (err) => {
        console.error('Failed to load simulations', err);
      }
    });
  }

  deleteSim(id: string) {
    this.simulationService.deleteSimulation(id).subscribe({
      next: () => {

        this.simulations = this.simulations.filter(sim => sim._id !== id);

        this.countSim.emit(this.simulations.length);
      },
      error: (err) => {
        console.error('Failed to delete simulation', err);
        alert('Delete failed');
      }
    });
  }
}
