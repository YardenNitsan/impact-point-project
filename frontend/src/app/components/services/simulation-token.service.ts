import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class SimulationTokenService {
  private readonly storageKey = 'simulationTokens';

  private readMap(): Record<string, string> {
    try {
      return JSON.parse(localStorage.getItem(this.storageKey) || '{}');
    } catch {
      return {};
    }
  }

  saveToken(simulationId: string, token: string): void {
    const map = this.readMap();
    map[simulationId] = token;
    localStorage.setItem(this.storageKey, JSON.stringify(map));
  }

  getToken(simulationId: string): string | null {
    const map = this.readMap();
    return map[simulationId] ?? null;
  }

  removeToken(simulationId: string): void {
    const map = this.readMap();
    delete map[simulationId];
    localStorage.setItem(this.storageKey, JSON.stringify(map));
  }

  clearAll(): void {
    localStorage.removeItem(this.storageKey);
  }
}
