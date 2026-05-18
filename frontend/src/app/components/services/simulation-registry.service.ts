import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export interface StoredSimulation {
  id: string;
  createdAt: string;
  durationSeconds: number;
  weather_source: 'machine' | 'api' | 'knn' | 'calculations';
}

@Injectable({ providedIn: 'root' })
export class SimulationRegistryService {
  private readonly storageKey = 'simulationRegistry';

  private readonly simulationsSubject = new BehaviorSubject<StoredSimulation[]>(
    this.readAndSort(),
  );

  readonly simulations$: Observable<StoredSimulation[]> =
    this.simulationsSubject.asObservable();

  private readList(): StoredSimulation[] {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return [];

      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  private sort(items: StoredSimulation[]): StoredSimulation[] {
    return [...items].sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    );
  }

  private readAndSort(): StoredSimulation[] {
    return this.sort(this.readList());
  }

  private publish(items: StoredSimulation[]): void {
    const sorted = this.sort(items);
    localStorage.setItem(this.storageKey, JSON.stringify(sorted));
    this.simulationsSubject.next(sorted);
  }

  getAll(): StoredSimulation[] {
    return this.simulationsSubject.value;
  }

  saveSimulation(simulation: StoredSimulation): void {
    const next = this.readList().filter((item) => item.id !== simulation.id);
    next.unshift(simulation);
    this.publish(next);
  }

  removeSimulation(id: string): void {
    this.publish(this.readList().filter((item) => item.id !== id));
  }

  refresh(): void {
    this.simulationsSubject.next(this.readAndSort());
  }

  clearAll(): void {
    localStorage.removeItem(this.storageKey);
    this.simulationsSubject.next([]);
  }
}
