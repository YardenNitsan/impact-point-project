import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SimulationHistoryService, SimulationHistoryItem } from './services/simulation-history.service';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent implements OnInit {

  simulations: SimulationHistoryItem[] = [];

  constructor(private historyService: SimulationHistoryService) {}

  ngOnInit(): void {
    this.historyService.getSimulations().subscribe({
      next: (data) => {
        this.simulations = data;
      },
      error: (err) => {
        console.error('Failed to load simulations', err);
      }
    });
  }
}
