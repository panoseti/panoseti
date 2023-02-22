/*
 * TMC2209 - 2A Stepper Motor Driver 
 *
 * Copyright (C)2020 Tom Lafleur
 *    tom@lafleur.us
 *
 * based on the work of: Laurentiu Badea
 *
 * This file may be redistributed under the terms of the MIT license.
 * A copy of this license has been included with this distribution in the file LICENSE.
 */
#ifndef TMC2209_H
#define TMC2209_H
#include <Arduino.h>
#include "BasicStepperDriver.h"

class TMC2209 : public BasicStepperDriver {
protected:
    short MS1 =     PIN_UNCONNECTED;
    short MS2 =     PIN_UNCONNECTED;
    short spread =  PIN_UNCONNECTED;
    short diag =    PIN_UNCONNECTED;          // Not use here, but for rederence only for future use
    
    // tWH(STEP) pulse duration, STEP high, min value
    static const int step_high_min = 0;   // 0.1us
    // tWL(STEP) pulse duration, STEP low, min value
    static const int step_low_min = 0;    // 0.1us
    // tWAKE wakeup time, nSLEEP inactive to STEP
    static const int wakeup_time = 1500;		// Unknown ??
    // also 200ns between ENBL/DIR/Mx changes and STEP HIGH

    // Get max microsteps supported by the device
    short getMaxMicrostep() override;

private:
    // microstep range (1, 16, 32 etc)
    static const short MAX_MICROSTEP = 64;

public:
    /*
     * Basic connection: only DIR, STEP are connected.
     * Microstepping controls should be hardwired.
     */
    TMC2209(short steps, short dir_pin, short step_pin);
    TMC2209(short steps, short dir_pin, short step_pin, short enable_pin);
    /*
     * DIR, STEP and microstep control MS1, MS2
     */
    TMC2209(short steps, short dir_pin, short step_pin, short MS1, short MS2);
    TMC2209(short steps, short dir_pin, short step_pin, short enable_pin, short MS1, short MS2);
     /*
     * Fully Wired - DIR, STEP, microstep, diag and spread
     */
    TMC2209(short steps, short dir_pin, short step_pin, short MS1, short MS2, short spread, short diag);
    TMC2209(short steps, short dir_pin, short step_pin, short enable_pin, short MS1, short MS2, short spread, short diag);

    void begin(float rpm=60, short microsteps=1);

    short setMicrostep(short microsteps) override;

    void setSpread(bool mode);
};
#endif // TMC2209_H
