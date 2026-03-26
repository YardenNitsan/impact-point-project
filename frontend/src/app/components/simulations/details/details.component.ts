import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnDestroy,
  Output,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  SimulationDetailsService,
  SimulationDetails,
} from './details-services/simulationDetails.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-details',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './details.component.html',
  styleUrl: './details.component.css',
})
export class DetailsComponent implements OnChanges, OnDestroy {
  @Input({ required: true }) simulationId!: string;
  @Output() back = new EventEmitter<void>();

  formattedDuration = '';
  details!: SimulationDetails;
  loading = true;
  private subscription?: Subscription;

  constructor(private detailsService: SimulationDetailsService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['simulationId'] && this.simulationId) {
      this.loadDetails();
    }
  }

  private loadDetails() {
    this.loading = true;

    this.subscription?.unsubscribe();

    this.subscription = this.detailsService
      .getDetails(this.simulationId)
      .subscribe({
        next: (data) => {
          this.formattedDuration = this.formatTime(data.durationSeconds);
          this.details = data;
          this.loading = false;
        },
        error: () => {
          alert('Failed to load simulation details');
          this.loading = false;
        },
      });
  }

  private formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  ngOnDestroy(): void {
    this.subscription?.unsubscribe();
  }
}
