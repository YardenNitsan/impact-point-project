import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnInit,
  Output,
  SimpleChanges
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  SimulationDetailsService,
  SimulationDetails
} from './details-services/simulationDetails.service';

@Component({
  selector: 'app-details',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './details.component.html',
  styleUrl: './details.component.css'
})
export class DetailsComponent implements OnInit {

  @Input({ required: true }) simulationId!: string;
  @Output() back = new EventEmitter<void>();

  details!: SimulationDetails;
  loading = true;

  constructor(private detailsService: SimulationDetailsService) {}

  ngOnInit(): void {
    this.detailsService.getDetails(this.simulationId).subscribe({
      next: (data) => {
        this.details = data;
        this.loading = false;
      },
      error: () => {
        alert('Failed to load simulation details');
        this.loading = false;
      }
    });
  }
}

