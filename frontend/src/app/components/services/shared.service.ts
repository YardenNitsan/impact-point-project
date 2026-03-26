import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { Coordinate } from '../models/coordinate.model';

// needed to be a singelton service who holds data of two components
@Injectable({ providedIn: 'root' })
export class SharedService {
  // simulation initialize
  private subject = new BehaviorSubject<Coordinate[]>([]);

  // given Observable for read only
  data$ = this.subject.asObservable();

  // read last known value without the need to wait for another output
  get snapshot(): Coordinate[] {
    return this.subject.value;
  }
  public lastSimulationDuration = 60;

  setDuration(seconds: number) {
    this.lastSimulationDuration = seconds;
  }

  // give BehaviorSubject new value
  setData(coords: Coordinate[]) {
    this.subject.next(coords);
  }

  clear() {
    this.subject.next([]);
  }
}
