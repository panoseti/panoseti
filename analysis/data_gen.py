#! /usr/bin/env python3

# generate synthetic image data consisting of
# - Poisson-distributed noise
# - pulses with given power, period, phase, duty cycle, and position
#
# parameters:
# --file_duration X  (how much data to generate, in sec)
# --noise_mean N
# --pulse_power N
# --pulse_period X
# --pulse_phase X
# --pulse_duty_cycle X (0..1)
# --pulse_position i j  (pulse is in pixel (i,j))
#    future: gaussian disk?  fake cherenkov?
#
# image-mode parameters are taken from data_config.json

import numpy, sys, math
sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

def make_data(
    data_config,
    file_duration,
    noise_mean,
    pulse_power,
    pulse_period,
    pulse_phase,
    pulse_duty_cycle,
    pulse_position
):
    t = 0
    frame_period = data_config['image']['integration_time_usec']/1e6
    f = open('data_gen.pff', 'wb')
    while t < file_duration:
        image = numpy.random.poisson(noise_mean, [32,32])
        x = math.fmod(t-pulse_phase, pulse_period)
        y = x/pulse_period
        if y < pulse_duty_cycle:
            image[pulse_position[0]][pulse_position[1]] += pulse_power
        pff.write_image_16_2(f, image)
        t += frame_period
    f.close()

if __name__ == "__main__":
    file_duration = .01
    noise_mean = 10
    pulse_power = 10
    pulse_period = 1
    pulse_phase = 0
    pulse_duty_cycle = 1e-2
    pulse_position = [5,5]
    argv = sys.argv
    i = 1
    while i < len(argv):
        if argv[i] == '--file_duration':
            i += 1
            file_duration = float(argv[i])
        elif argv[i] == '--noise_mean':
            i += 1
            noise_mean = int(argv[i])
        elif argv[i] == '--pulse_power':
            i += 1
            nsec = int(argv[i])
        elif argv[i] == '--pulse_period':
            i += 1
            nsec = float(argv[i])
        elif argv[i] == '--pulse_phase':
            i += 1
            nsec = float(argv[i])
        elif argv[i] == '--pulse_duty_cycle':
            i += 1
            nsec = float(argv[i])
        elif argv[i] == '--pulse_position':
            i += 1
            pulse_position[0] = int(argv[i])
            i += 1
            pulse_position[1] = int(argv[i])
        else:
            raise Exception('unknown arg %s'%argv[i])
        i += 1
    data_config = config_file.get_data_config()
    make_data(
        data_config, file_duration, noise_mean,
        pulse_power, pulse_period, pulse_phase,
        pulse_duty_cycle, pulse_position
    )
