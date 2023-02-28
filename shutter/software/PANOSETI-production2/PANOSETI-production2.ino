
/**********************************************************************
   PANOSETI telescope shutter controller
   production version of board 6/8/2020
************************************************************************
  CHANGE LOG:

    DATE         REV  DESCRIPTION
    -----------  ---  ----------------------------------------------------------
    8-Dec-2019 1.0b  TRL - First Build of M0 version
    6/8/2020 production1 FA begin production version
    6/21/2020 production1 FA changed command to match documentation
    6/23/2020 added the status bits returned to outside world
    9/8/2020 moved powerdown signal to arduino D21
    9/16/2021 changed day/night threshold

   (c) 2019, Tom Lafleur tom@lafleur.us, Franklin Antonio franklin@franklinantonio.org

*/

/* ************************************************************* */
// Optional features that can be enabled

#define __DEBUG1__                                            // Enable Debug printing....
#define __DEBUG2__

#define MyWDT                                               // if using the Watch Dog Timer  

/* ************************************************************* */
//#define DemoBoard1a_DRV8880                                   // this is the first version of the PWB board with DRV8880
//#define DemoBoard2_TMC2209                                    // this is the 2nd version of the PWB board with StepStick socket
//#define DemoBoard2_DRV8880                                    // demo board wtih DRV8800 driver chip
#define productionboard1                                        // production PC board

/* ************************************************************* */
#define SKETCHNAME      "PANOSETI Shutter Controller"
#define SKETCHVERSION   "3.0"


/* ************************************************************* */
#ifdef __DEBUG1__
#define    debug1(f,...)    DebugPrint(f, ##__VA_ARGS__);
#else
#define    debug1(f,...)
#endif

#ifdef __DEBUG2__
#define    debug2(f,...)    DebugPrint(f, ##__VA_ARGS__);
#else
#define    debug2(f,...)
#endif


/* ************************************************************* */
// Select processor includes
#include <Arduino.h>
#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>

//#include "DRV8880.h"                // https://github.com/laurb9/StepperDriver
#include "TMC2209.h"

#ifdef MyWDT
#include "WDTZero.h"                // https://github.com/javos65/WDTZero
WDTZero MyWatchDog;                 // Define WDT for watch-dog timer support
#endif


/* ************************************************************* */
#define DAY         86400000   // 24 hr in ms
#define HOUR        3600000    // 1 hr in ms
#define MINUTE      60000      // 1 Minute in ms
#define SECOND      1000       // 1000 ms
/* ************************************************************* */

/* ***************** DemoBoard 1a with DRV8880 ******************** */
#ifdef DemoBoard1a_DRV8880
// Define stepper driver pins for PCB Version 1 and 1a

// Motor steps per revolution. Most steppers are 200 steps or 1.8 degrees/step
#define MOTOR_STEPS   200
// Target RPM for cruise speed
#define RPM           15
// Acceleration and deceleration values are always in FULL steps / s^2
#define MOTOR_ACCEL   2000
#define MOTOR_DECEL   1000

// Microstepping mode. If you hardwired it to save pins, set to the same value here.
#define MICROSTEPS    16

// define motor controller pins
#define DIR           8
#define STEP          9
#define SLEEP         7               // optional (just delete SLEEP from everywhere if not used)

#define M0            10
#define M1            11
#define TRQ0          12
#define TRQ1          6

#define Enable        5
#define Fault         4

// enable the stepper driver
DRV8880 stepper(MOTOR_STEPS, DIR, STEP, SLEEP, M0, M1, TRQ0, TRQ1);

// define other I/O pins

#define LimitCW_Pin    2      
#define LimitCCW_Pin   15

#define LimitReach     1                  // set polarity of limit bit: 1 = reach limit, 0 = ok, we have an inversion on pin
#define LimitOK        0

#define PhotoCellPin   A5
#define DACOutPin      A0 
#define ledPin         13                  // Error Led

#define ExtIn0Pin      0
#define ExtIn1Pin      1
#define ExtOut0Pin     3 
#define ExtOut1Pin     8
/* ************************************************************* */



/* ***************** DemoBoard 2 with TMC2209 ******************** */
#elif defined DemoBoard2_TMC2209
// Define stepper driver pins for PCB Version 2 board with StepStick

// Motor steps per revolution. Most steppers are 200 steps or 1.8 degrees/step
#define MOTOR_STEPS   400
// Target RPM for cruise speed
#define RPM           4 
// Acceleration and deceleration values are always in FULL steps / s^2
#define MOTOR_ACCEL 2000
#define MOTOR_DECEL 1000

// Microstepping mode. If you hardwired it to save pins, set to the same value here.
#define MICROSTEPS 8

#define DIR           8
#define STEP          9
#define MS1           11      // F3
#define MS2           10      // F2
#define spread        12      // F1
#define uart          6       // F0
#define pdn           21       // 
#define Enable        5
#define diag          PIN_UNCONNECTED
#define Fault         4

// enable the stepper driver
TMC2209 stepper(MOTOR_STEPS, DIR, STEP, Enable, MS1, MS2, spread, diag);

// define other I/O pins

#define LimitCW_Pin    2      
#define LimitCCW_Pin   15

#define LimitReach     1                  // set polarity of limit bit: 1 = reach limit, 0 = ok, we have an inversion on pin
#define LimitOK        0

#define PhotoCellPin   A5
#define DACOutPin      A0 
#define ledPin         13                  // Error Led

#define ExtIn0Pin      0
#define ExtIn1Pin      1
#define ExtOut0Pin     3 
#define ExtOut1Pin     8
/* ************************************************************* */



/* ***************** DemoBoard 2 with DRV8880 ******************** */
#elif defined DemoBoard2_DRV8880
// Define stepper driver pins for PCB Version 2 board with StepStick

// Motor steps per revolution. Most steppers are 200 steps or 1.8 degrees/step
#define MOTOR_STEPS   200
// Target RPM for cruise speed
#define RPM           10
// Acceleration and deceleration values are always in FULL steps / s^2
#define MOTOR_ACCEL   2000
#define MOTOR_DECEL   1000

// Microstepping mode. If you hardwired it to save pins, set to the same value here.
#define MICROSTEPS    16

#define DIR           8
#define STEP          9
#define M1            11      // F3
#define M0            10      // F2
#define TRQ1          12      // F1
#define TRQ0          6       // F0
#define SLEEP         7       // F4
#define Enable        5
#define Fault         4

// enable the driver
DRV8880 stepper(MOTOR_STEPS, DIR, STEP, SLEEP, M0, M1, TRQ0, TRQ1);

// define other I/O pins

#define LimitCW_Pin    2      
#define LimitCCW_Pin   15

#define LimitReach     1                  // set polarity of limit bit: 1 = reach limit, 0 = ok, we have an inversion on pin
#define LimitOK        0

#define PhotoCellPin   A5
#define DACOutPin      A0 
#define ledPin         13                  // Error Led

#define ExtIn0Pin      0
#define ExtIn1Pin      1
#define ExtOut0Pin     3 
#define ExtOut1Pin     8
/* ************************************************************* */



/* ***************** production PC board ******************** */
#elif defined productionboard1
// production PC board 

// Motor steps per revolution. Most steppers are 200 steps or 1.8 degrees/step
#define MOTOR_STEPS   200
// Target RPM for cruise speed
#define RPM           12 
// Acceleration and deceleration values are always in FULL steps / s^2
#define MOTOR_ACCEL 2000
#define MOTOR_DECEL 1000

// Microstepping mode. If you hardwired it to save pins, set to the same value here.
#define MICROSTEPS 32

#define DIR           8
#define STEP          9
#define MS1           11      // F3
#define MS2           10      // F2
#define spread        12      // F1
#define uart          6       // F0
#define pdn           21      // rev 3a of board has pdn on D21, ie cpu pin 32
#define Enable        5
#define diag          PIN_UNCONNECTED
#define Fault         4

// enable the stepper driver
TMC2209 stepper(MOTOR_STEPS, DIR, STEP, Enable, MS1, MS2, spread, diag);

// define other I/O pins

#define LimitCW_Pin    15                   //close
#define LimitCCW_Pin   2                  //open

#define LimitReach     1                  // set polarity of limit bit: 1 = reach limit, 0 = ok, we have an inversion on pin
#define LimitOK        0

#define PhotoCellPin   A5
#define DACOutPin      A0 
#define ledPin         13                  // Error Led

#define CommandInPin   18
#define ExtIn1Pin      3
#define ExtOut0Pin     0 
#define ExtOut1Pin     1
/* ************************************************************* */

/* ************************************************************* */
#else
#error ---->  Stepper Board NOT defined  <----
#endif
/* ************************************************************* */




/* ************************************************************* */
// system defines
#define CW             1                    // direction of travel...
#define CCW           -1

#define low            0
#define high           1

// use by LED function
#define off            0
#define on             1
#define toggle         2


#define day            350                  // ATD count use to define when we switch between day and night
#define night          400
// we use hysteresis with the above limits.
// my home workshop reads about 1000, but is dimmer than many school labs
// 6/23/2020 I did an outdoor test in thick "June gloom" and it came in 102 to 167.  This gives some confidence that open sky will be <100.  
// We only need to close the shutter when there is a risk of direct sunlight which could be focused by the lens.  Open sky contains the sun (somewhere)
// so that's the condition which should always close.  Range should be large enough so that the act of opening or closing the shutter doesn't change the
// light level enough to make it oscillate between open and closed, in any routine situation.
// 08/2021  We changed the pullup resistor value, so numbers are different.  Total dark 1000, pointing right at sun 100, ...

#define DebounceTime    35                  // time in millisec, used in ISR

/* ************************************************************* */
// Select correct defaults for the processor and board we are using
#ifdef __SAMD21G18A__                       // Using an ARM SAMD21G18A
#else
#error -------> Processor not defined, Requires an ARM M0 SAMD21G18A <-------
#endif


/* ************************************************************* */
// Global variables....
char msg1[80];                              // char string buffer (we always need one...)

const char compile_file[]  = __FILE__ ;
const char compile_date[]  = __DATE__ ", " __TIME__;
const char copyright[] = "(c) 2016-2020,  Tom Lafleur\n\n";

bool daylight = true;
bool presentdirection=true;   //FA
bool directionnotestablished=true;

/* ************************************************************* */
// These are used in the interrupt service routine so they may need be volatile
static bool AtLimitCW   = false;            // current state of limit switch CW
static bool AtLimitCCW  = false;            // current state of limit switch CCW

volatile static bool LimitCW   = false;     // Set in ISR
volatile static bool LimitCCW  = false;     // Set in ISR


/* ****************** Forward Declaration ********************* */
void setup ();
void loop ();
int FreeRam ();
void DebugPrint(const char *fmt, ... );
void ExtIn0_ISR();
void ExtIn1_ISR();
void Fault_ISR();
void LimitCW_ISR();
void LimitCCW_ISR();
void ErrorLED (int state);
int  ReadATD();
void WDT_Trigger();


/* ************************************************************* */
// Will return amount of free ram in M0 processor...
extern "C" char *sbrk(int i);
int FreeRam ()
{
  char stack_dummy = 0;
  return &stack_dummy - sbrk(0);
}


/* ************************************************************* */
// 0 = off, 1 = on, 2 = toggle state
void ErrorLED (int state)
{
  if ( state == 0) digitalWrite(ledPin, 0);                     // turn off LED
  if ( state == 1) digitalWrite(ledPin, 1);                     // turn on LED
  if ( state == 2) digitalWrite(ledPin, !digitalRead(ledPin));  // toggle it
}


/* ************************************************************* */
// This is used to emulate the printf function that is not avilable
// in many Arduino ports, like for the ARM M0 processor
void DebugPrint(const char *fmt, ... )
{
  char fmtBuffer[256];
  va_list args;
  va_start (args, fmt );
  va_end (args);
  vsnprintf(fmtBuffer, sizeof(fmtBuffer) - 1, fmt, args);
  va_end (args);
  Serial.print(fmtBuffer);
  Serial.flush();
}


/* ************************************************************* */
// Interrupt service routines.... 
// NOTE: debug print statement are not a good thing here....
// Any variable use in ISR should be define as volatile
/* ************************************************************* */
void Fault_ISR()
{
  debug1("** In Fault ISR\n" );
}


/* ************************************************************* */
void ExtIn0_ISR()
{
  debug1("** In Ext In-0 ISR\n" );
}


/* ************************************************************* */
void ExtIn1_ISR()
{
  debug1("** In Ext In-1 ISR\n" );
}


// In this interrupt we will debounce the mechanical switch...
/* ************************************************************* */
void LimitCW_ISR()
{
  volatile static unsigned long PreviousLimitCW_interrupt_time = 0;

  unsigned long interrupt_time = millis();
  if (interrupt_time - PreviousLimitCW_interrupt_time >= DebounceTime)  // ignores interupts for DebounceTime in milliseconds
  {
    debug1("** In CW Limit ISR: %d\n", digitalRead(LimitCW_Pin));
    LimitCW = true;   // ( we need to reset this flag when we respond to interrupt...)
  }
  PreviousLimitCW_interrupt_time = interrupt_time;
}


/* ************************************************************* */
void LimitCCW_ISR()
{
  volatile static unsigned long PreviousLimitCCW_interrupt_time = 0;

  unsigned long interrupt_time = millis();
  if (interrupt_time - PreviousLimitCCW_interrupt_time >= DebounceTime)  // ignores this interupts for DebounceTime in milliseconds
  {
    debug1("** In CCW Limit ISR: %d\n", digitalRead(LimitCCW_Pin) );
    LimitCCW = true;    // ( we need to reset this flag when we respond to interrupt...)
  }
  PreviousLimitCCW_interrupt_time = interrupt_time;
}


/* ************************************************************* */
#define NumSamples 32
int ReadATD()
{
  int count  = analogRead (PhotoCellPin) ;                     // this is a junk read to clear ATD
  delay (10);
  
  count = 0;                                            // we will take multiple reading to get a stable count
  for (int i = 0; i < NumSamples; i++)  count += analogRead(PhotoCellPin);
  count /= NumSamples;
  return count;
}


/* ************************************************************* */
// assumes a 12 bit DAC
void setDAC0(int value)
{
  constrain(value, 0, 4095);
  analogWrite(DACOutPin, value);   // DAC-0 output
}


/* ************************************************************* */
// Watchdog interrupt routine.

#ifdef  MyWDT
void WDT_Trigger()
{
  debug1("** WDT has triggered...");
  delay(2000);
  NVIC_SystemReset();                        // processor will do a software reset
}
#endif

int DirFlag = CW;        // 1=CW, -1=CCW



/* ************************************************************* */
/* ************************* Setup ***************************** */
/* ************************************************************* */
void setup()
{
  asm(".global _printf_float");             // this forces the compiler to allow floating point in printf (debug1..)

  int t = 20; //Initialize serial and wait for port to open, max 10 second
  Serial.begin(115200);                     // setup serial port-0
  while (!Serial)
  { delay(500);if ( (t--) == 0 ) break; }

  debug1("** %s %s\n", SKETCHNAME, SKETCHVERSION);
  debug1("** %s \n",   compile_file);
  debug1("** %s \n", compile_date);
  debug1("** Free Ram: %d bytes\n\n", FreeRam ());
  debug1("** Day/Nightthresholds: %u, %u\n", day, night);

#if   defined DemoBoard1a_DRV8880
  debug1("\n*** Using Demo-Board 1a with DRV8880\n\n");
#elif defined  DemoBoard2_TMC2209
  debug1("\n*** Using Demo-Board 2 with TMC2209\n\n");
#elif defined  DemoBoard2_DRV8880
  debug1("\n*** Using Demo-Board 2 with DRV8880\n\n");
#elif defined productionboard1
  debug1("\n*** Using production board version 1 with TMC2209 chip\n\n");
#else
  debug1("\n*** Demo-Board not defined!!!\n");
#endif

// set up ATD and reference, for ATD to use:
// options --> AR_DEFAULT, AR_INTERNAL, AR_EXTERNAL, AR_INTERNAL1V0, AR_INTERNAL1V65, AR_INTERNAL2V23
  analogReference(AR_DEFAULT);              // AR_DEFAULT is set to VDD -> +3.3v, AR_EXTERNAL is set to external pin on chip
  analogReadResolution(10);                 // we want 10 bits
  // set up DAC
  analogWriteResolution(12);
  analogWrite(DACOutPin,400);               // set motor current
  // motor current is set by the Vref pin of the TMC2209.  Vref range is 0.5v to 2.5v.  
  // the DAC has a range of 0v to 3.3v, corresponding to numerical values 0 to 1023
  // I could have set up a formula here, but that would be worthless, because in practice I set the motor current by measuring power and
  // adjusting the number by trial and error. 

  // Motor current is hard to measure because of microstepping and pulse-width-modulation.  Most meters can't produce a useful measurement.
  // to get around this, I avoid measuring current in the motor windings directly.  Instead, I measure current on the +24V input line.  
  // There any meter will give you a correct reading, because the current is not pulsed.  Almost all the power
  // goes to the motor, so just measure the input current and multiply by 24v to get watts to the motor.  The motors are specified for a certain current
  // which is 220mA for our motor.  Using this and the motor winding resistance, you can compute the spec power level by I^2 R.  When you get that
  // power level on the input, you've chosen the right value for the DAC.  Of course you can run the motor at less current, which runs cooler, as long as it
  // produces enough torque, you're good.

  
#ifdef  MyWDT
// Watch Dog has a hard and soft-watchdog, see github documents  
// Hard WD can be set from 62ms upto 16 seconds response
// Soft WD for longer time....
  MyWatchDog.attachShutdown(WDT_Trigger);
  MyWatchDog.setup(WDT_HARDCYCLE16S);      // initialize WDT-Hard refesh cycle on 16sec interval  
//  MyWatchDog.setup(WDT_SOFTCYCLE32S);    // initialize WDT-softcounter refesh cycle on 32sec interval
#endif

  // Lets set up I/O pins
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, 0);                 // turn off LED

  pinMode(ExtOut0Pin, OUTPUT);
  digitalWrite(ExtOut0Pin, 0);             // turn them off
  pinMode(ExtOut1Pin, OUTPUT);
  digitalWrite(ExtOut1Pin, 0);
                 
  pinMode(CommandInPin, INPUT);
  pinMode(ExtIn1Pin, INPUT); 
  pinMode(Fault, INPUT);
  
  pinMode(LimitCW_Pin, INPUT);
  pinMode(LimitCCW_Pin, INPUT);

  pinMode(pdn, OUTPUT);                   //FA
  digitalWrite(pdn, LOW);              //FA this enables the powerdown function in the motor driver chip
               
// Lets setup stepper driver
  stepper.begin( RPM, MICROSTEPS);

#if defined DemoBoard2_TMC2209 || defined productionboard1
// if using enable/disable on ENABLE pin (active LOW) instead of SLEEP uncomment next line
  stepper.setEnableActiveState(LOW);
  stepper.enable();
#elif defined DemoBoard1a_DRV8880 || defined DemoBoard2_DRV8880
  stepper.setEnableActiveState(HIGH);
  stepper.enable();
#endif

// set current level (for DRV8880 only). Valid percent values are 25, 50, 75 or 100.
//#if  DemoBoard1a_DRV8880 || DemoBoard2_DRV8880
//  stepper.setCurrent(100);
//#endif

// Set LINEAR_SPEED (accelerated) profile.  CONSTANT_SPEED or LINEAR_SPEED
//  stepper.setSpeedProfile(stepper.LINEAR_SPEED, MOTOR_ACCEL, MOTOR_DECEL);
    stepper.setSpeedProfile(stepper.CONSTANT_SPEED);

  DirFlag = CW;
  stepper.enable();                                       // make sure motor is powered on
  presentdirection=true;                                  // set initial direction (to close shutter by default,eh?)
//  stepper.startMove(-10000*MICROSTEPS);                   // initially move to close shutter
//  debug1("** Starting Stepper in a CW direction...\n\n");
  
// setup for interrupt's...
  noInterrupts();
  
// options are--> LOW, HIGH, CHANGE, RISING, FALLING
//    attachInterrupt(digitalPinToInterrupt(ExtIn0Pin), ExtIn0_ISR, RISING);
//    attachInterrupt(digitalPinToInterrupt(ExtIn1Pin), ExtIn1_ISR, RISING);
    attachInterrupt(digitalPinToInterrupt(Fault), Fault_ISR, FALLING);      // ? <----- neeed to check direct of action
//    attachInterrupt(digitalPinToInterrupt(LimitCW_Pin), LimitCW_ISR, CHANGE);
//    attachInterrupt(digitalPinToInterrupt(LimitCCW_Pin), LimitCCW_ISR, CHANGE);

  interrupts();                         // always make this the last thing...
}   // end of setup


/* ************************************************************* */
// Simple task scheduler variable's
// Task 1
#define  Task1Time (SECOND * 1)
unsigned long Task1lastSendTime        = 0;
unsigned long Task1Count               = 0;

// Task 2
#define  Task2Time (SECOND * 30)
unsigned long Task2lastSendTime        = 0;
unsigned long Task2Count               = 0;

// Task 3
#define  Task3Time (500)
unsigned long Task3lastSendTime        = 0;

// Task 4, this task uses usec timer --> micros()
#define  Task4Time (500)               // usec
unsigned long Task4lastSendTime        = 0;

// Loop Delay
#define  DelayTime (SECOND * 5)
unsigned long DelaylastSendTime        = 0;

/* ************************************************************* */
/* ************************* LOOP ****************************** */
/* ************************************************************* */

void loop()
{
// arduino's nonpreemptive scheduling loop.  Nonblocking actions only please.
  bool commanddirection;
  bool desireddirection;

// test of the watchdog.
//if(digitalRead(ExtIn1Pin)) {
//  debug1("triggering an infinite loop now to test watchdog...\n");
//  while(1);
//  }

  
  commanddirection = digitalRead(CommandInPin);                  // read the direction command pin
  desireddirection = commanddirection;
  if(daylight) desireddirection=false;
// the board has an inverter between the command input and the processor, so we read 0 to mean open the shutter

// if we call startMove with too large a number, internal overflow will foul us up, so we use a moderate number, and
// restart the motor if it stops.  Simple workaround of driver limititions.

  if(desireddirection != presentdirection || directionnotestablished) {     // are we already going right direction?
    debug1("desired direction changed to =%d\n",desireddirection); 
    stepper.stop();                                           // no, not going right direction, so stop
    directionnotestablished=false;                            // now we know which way we're going
    stepper.enable();                                         //because we might have disabled it at shutter close
//    long stepstomove = 10000*MICROSTEPS;
//    if(desireddirection==true) stepstomove = -stepstomove;
//    stepper.startMove(stepstomove);                           //begin moving in the correct direction
    long moveangle = 720;
    if(desireddirection==true) moveangle=-moveangle;
    stepper.startRotate(moveangle);
  }
  presentdirection=desireddirection;                          // now that we have possibly reversed,
                                                              // our present direction is the desired one

  long wait_time_micros = stepper.nextAction();               // this causes the next step of the stepper to be executed if needed...

  if(wait_time_micros >0) {                                   // if motor is moving, we might need to stop it at limit switches
// limit switch actions
    if(presentdirection==true && digitalRead(LimitCW_Pin)) {  // check if the shutter is open or not
      stepper.stop();
      
      stepper.enable();
      long deg = -10;
      stepper.startRotate(deg);
      stepper.stop();
      
      debug1("stopped at CW limit\n");
    }
    if(presentdirection==false && digitalRead(LimitCCW_Pin)) { // check if the shutter is closed or not
      stepper.stop();
      /*
      stepper.enable();
      long deg = 10;
      stepper.startRotate(deg);
      stepper.stop();
      */ 
//      stepper.disable();                                    // should we disable motor in the shutter closed state?  Dunno.
// I choose to leave motor enabled, because we've turned on the feature that reduces motor current when stopped, so power consumption is low.
      debug1("stopped at CCW limit\n");
    }

  bool atlimit;
  atlimit = presentdirection==true && digitalRead(LimitCW_Pin) || presentdirection==false && digitalRead(LimitCCW_Pin);
  digitalWrite(ExtOut0Pin, atlimit);                          // tell the world we're at our commanded position
  
  } 

  
 
/* ************************************************************* */  
// very simple task scheduler...
// it gets current time and check to see if our timer has expired
// if so, it time to do the task and then we restart task time to repeat...
  
// Task 1
  if (millis() - Task1lastSendTime >= Task1Time)         // do work for task 1
  {
    Task1Count ++;
    unsigned int PhotoCell = ReadATD();                 // get reading from ATD
    if (PhotoCell >= night) daylight = false;           // do we need to be concern about mid-range??
    if (PhotoCell <= day) daylight = true;
    digitalWrite(ExtOut1Pin,daylight);                  // let computer know light sensor status

    if(Task1Count%10==0)                                // occasionally report to serial monitor screen, so we can see the light level numbers
    {    
      debug1("** Task1 Count: %u, ATD: %u  Thresholds: %u %u\n", Task1Count, PhotoCell, day, night);
    }
    
//    if (daylight)
//    {
//      debug1("** It's Daylight...\n");
//    }
//    else
//    {
//      debug1("** It's Night time...\n");
//    }
    
    Task1lastSendTime = millis();
  } 

// Task 2
    if (millis() - Task2lastSendTime >= Task2Time)         // do work for task 2
    {
      Task2Count ++;
//      debug1("** Task2 Count: %u\n", Task2Count);
      Task2lastSendTime = millis();
    }

// Task 3
    if (millis() - Task3lastSendTime >= Task3Time)         // do work for task 3
    {
      ErrorLED (toggle);
      Task3lastSendTime = millis();
    }

  
// Task 4, this task is using micros() for shorter response task...
    if (micros() - Task4lastSendTime >= Task4Time)         // do work for task 4
    {
      Task4lastSendTime = micros();
    }


#ifdef MyWDT
  MyWatchDog.clear();                                        // refresh wdt - before it times out...
#endif

}   // end of loop
/* ************************************************************* */
/* ************************************************************* */
