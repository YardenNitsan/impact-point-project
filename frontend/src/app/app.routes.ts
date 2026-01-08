import { Component } from '@angular/core';
import { Routes } from '@angular/router';
import { PageNotFoundComponent } from './components/page-not-found/page-not-found.component';
import { AppComponent } from './app.component';
import { HomeComponent } from './components/home/home.component';

export const routes: Routes = [
    {path: '', component: HomeComponent, pathMatch: 'full'},

    {path: '**', component: PageNotFoundComponent}
];
