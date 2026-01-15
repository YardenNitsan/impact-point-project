import { Injectable } from "@angular/core";
import { BehaviorSubject } from "rxjs";

export interface Coordinate {
  lon: number;
  lat: number;
  alt: number;
}

//needed to be a singelton service who holds data of two components
@Injectable({ providedIn: "root" })
export class SharedService {

  //there was no simulation initialize yet so will be defined to null
  private subject = new BehaviorSubject<Coordinate[] | null>(null);

  //given Observable for read only 
  data$ = this.subject.asObservable();

  //read last known value without the need to wait for another output
  get snapshot(): Coordinate[] | null {
    return this.subject.value;
  }

  //give BehaviorSubject new value
  setData(coords: Coordinate[]) {
    this.subject.next(coords);
  }

  clear() {
    this.subject.next(null);
  }
}
