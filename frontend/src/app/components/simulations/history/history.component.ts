import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SimulationHistoryService, SimulationHistoryItem } from './history-services/simulationHistory.service';
import { Coordinate, SharedService } from '../../services/shared.service';
import { Subject, Subscriber, Subscription, takeUntil } from 'rxjs';
import { Router } from '@angular/router';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent implements OnInit {

  private destroyed$: Subject<boolean> = new Subject();
  protected simulations: SimulationHistoryItem[] = [];
  @Output() countSim = new EventEmitter<number>();
  private cardsToNotScroll = 5;
  private shouldScroll = false;

  constructor(
    private simulationService: SimulationHistoryService,
    private shared: SharedService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.simulationService.getSimulations().pipe(takeUntil(this.destroyed$)).subscribe({
      next: (data) => {
        console.log(data);
        this.simulations = data;
        this.shouldScroll = data.length > this.cardsToNotScroll;
        this.countSim.emit(this.simulations.length);
      },
      error: (err) => {
        console.error('Failed to load simulations', err);
      }
    });
  }

  deleteSim(id: string) {
    console.log(id);
    this.simulationService.deleteSimulation(id).pipe(takeUntil(this.destroyed$)).subscribe({
      next: () => {

        this.simulations = this.simulations.filter(sim => sim.id !== id);

        this.countSim.emit(this.simulations.length);
      },
      error: (err) => {
        console.error('Failed to delete simulation', err);
        alert('Delete failed');
      }
    });
  }
  watchSim(id: string) {
  this.simulationService.watchSimulation(id)
    .subscribe({
      next: (sim: any) => {
        console.log('WATCH RESPONSE:', sim);

        //save coordinates only cause other are not important here
        this.shared.setData(sim);

        this.router.navigateByUrl('/');
      },
      error: () => alert('load failed')
    });
  }


  ngOnDestroy(){
    this.destroyed$.next(true);
    this.destroyed$.complete();
  }
}