/*
 * TMC2209 - 2A Stepper Motor Driver 
 *
 * Copyright (C)2020 Tom Lafleur
 *		tom@lafleur.us
 *
 * based on the work of: Laurentiu Badea
 *
 * This file may be redistributed under the terms of the MIT license.
 * A copy of this license has been included with this distribution in the file LICENSE.
 */
#include "TMC2209.h"

/*
 * Basic connection: only DIR, STEP are connected.
 * Microstepping controls should be hardwired.
 */
TMC2209::TMC2209(short steps, short dir_pin, short step_pin)
:BasicStepperDriver(steps, dir_pin, step_pin)
{}

TMC2209::TMC2209(short steps, short dir_pin, short step_pin, short enable_pin)
:BasicStepperDriver(steps, dir_pin, step_pin, enable_pin)
{}

/*
 * All the necessary control pins plus microstepping for TMC2209 are connected.
 */
TMC2209::TMC2209(short steps, short dir_pin, short step_pin, short MS1, short MS2)
:BasicStepperDriver(steps, dir_pin, step_pin), MS1(MS1), MS2(MS2)
{}

TMC2209::TMC2209(short steps, short dir_pin, short step_pin, short enable_pin, short MS1, short MS2)
:BasicStepperDriver(steps, dir_pin, step_pin, enable_pin), MS1(MS1), MS2(MS2)
{}

/*
 * Fully wired. All control pins for TMC2209 are connected.
 */
TMC2209::TMC2209(short steps, short dir_pin, short step_pin, short MS1, short MS2, short spread, short diag)
:BasicStepperDriver(steps, dir_pin, step_pin), MS1(MS1), MS2(MS2), spread(spread), diag(diag)
{}

TMC2209::TMC2209(short steps, short dir_pin, short step_pin, short enable_pin, short MS1, short MS2, short spread, short diag)
:BasicStepperDriver(steps, dir_pin, step_pin, enable_pin), MS1(MS1), MS2(MS2), spread(spread), diag(diag)
{}

void TMC2209::begin(float rpm, short microsteps){
    BasicStepperDriver::begin(rpm, microsteps);
    pinMode(spread, OUTPUT);
    digitalWrite(spread, false);    // default to StealthChop
}

short TMC2209::getMaxMicrostep(){
    return TMC2209::MAX_MICROSTEP;
}

/*
 * Set microstepping mode (1:divisor)
 * Allowed ranges for TMC2209 are 1:8 to 1:64
 * If the control pins are not connected, we recalculate the timing only
 */
short TMC2209::setMicrostep(short microsteps){
    BasicStepperDriver::setMicrostep(microsteps);

    if (!IS_CONNECTED(MS1) || !IS_CONNECTED(MS2)){
        return this->microsteps;
    }

    /*
     * Step mode truth table
     * MS2 MS1  step mode
     *  0  0     8
     *  1  1     16
     *  0  1     32
     *  1  0     64
     *
     */

    pinMode(MS2, OUTPUT);
    pinMode(MS1, OUTPUT);
    switch(this->microsteps){
        case 8:
            digitalWrite(MS2, LOW);
            digitalWrite(MS1, LOW);
            break;
        case 16:
            digitalWrite(MS2, HIGH);
            digitalWrite(MS1, HIGH);
            break;
        case 32:
            digitalWrite(MS2, LOW);
            digitalWrite(MS1, HIGH);
            break;
        case 64:
            digitalWrite(MS2, HIGH);
            digitalWrite(MS1, LOW);
            break;
    }
    return this->microsteps;
}

void TMC2209::setSpread(bool mode){
    /*
     * 0 false = StealthChop
     * 1 true  = SpreadCycle
  
     */
    if (!IS_CONNECTED(spread) ) {
        return;
    }    
    pinMode(spread, OUTPUT);
    digitalWrite(spread, mode);

}
