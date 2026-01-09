import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PreviousSimulationsComponent } from './previous-simulations.component';

describe('PreviousSimulationsComponent', () => {
  let component: PreviousSimulationsComponent;
  let fixture: ComponentFixture<PreviousSimulationsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PreviousSimulationsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PreviousSimulationsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
