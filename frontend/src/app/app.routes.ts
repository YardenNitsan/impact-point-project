import { Routes } from '@angular/router';
import { PageNotFoundComponent } from './components/page-not-found/page-not-found.component';
import { HomeComponent } from './components/home/home.component';
import { SimulationsComponent } from './components/simulations/simulations.component';

export const routes: Routes = [
    {path: '', component: HomeComponent, pathMatch: 'full'},
    {path: 'sims', component: SimulationsComponent},
    {path: '**', component: PageNotFoundComponent}
];
