import {
  Component,
  EventEmitter,
  OnInit,
  Output,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  SimulationHistoryService,
  SimulationHistoryItem,
} from './history-services/simulationHistory.service';
import { SharedService } from '../../services/shared.service';
import { Subject, takeUntil } from 'rxjs';
import { Router } from '@angular/router';
import { Coordinate } from '../../models/coordinate.model';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css',
})
export class HistoryComponent implements OnInit, OnDestroy {
  private destroyed$: Subject<void> = new Subject();
  protected simulations: SimulationHistoryItem[] = [];
  private cardsToNotScroll = 5;
  private shouldScroll = false;

  @Output() countSim = new EventEmitter<number>();
  @Output() openDetails = new EventEmitter<string>();

  constructor(
    private simulationService: SimulationHistoryService,
    private shared: SharedService,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.simulationService
      .getSimulations()
      .pipe(takeUntil(this.destroyed$))
      .subscribe({
        next: (data) => {
          console.log(data);

          this.simulations = data.map((sim) => ({
            ...sim,
            formattedDuration: this.formatTime(sim.durationSeconds),
          }));

          this.shouldScroll = data.length > this.cardsToNotScroll;
          this.countSim.emit(this.simulations.length);
        },
        error: (err) => {
          console.error('Failed to load simulations', err);
        },
      });
  }

  deleteSim(id: string) {
    this.simulationService
      .deleteSimulation(id)
      .pipe(takeUntil(this.destroyed$))
      .subscribe({
        next: () => {
          this.simulations = this.simulations.filter((sim) => sim.id !== id);
          this.shouldScroll = this.simulations.length > this.cardsToNotScroll;
          this.countSim.emit(this.simulations.length);
        },
        error: (err) => {
          console.error('Failed to delete simulation', err);
          alert(
            err?.message ||
              err?.error?.error?.message ||
              'Failed to delete simulation',
          );
        },
      });
  }

  watchSim(id: string) {
    this.simulationService
      .watchSimulation(id)
      .pipe(takeUntil(this.destroyed$))
      .subscribe({
        next: (coords: Coordinate[]) => {
          console.log('WATCH RESPONSE:', coords);
          this.shared.setData(coords);
          this.router.navigateByUrl('/');
        },
        error: (err) =>
          alert(
            err?.message ||
              err?.error?.error?.message ||
              'Failed to load simulation',
          ),
      });
  }

  formatTime(seconds: number): string {
    if (!seconds) return '0:00';

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  details(id: string) {
    this.openDetails.emit(id);
  }

  ngOnDestroy() {
    this.destroyed$.next();
    this.destroyed$.complete();
  }
}
