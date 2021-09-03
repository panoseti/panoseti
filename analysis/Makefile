DEBUG = 1

PROGS = pulse write_images

all: $(PROGS)

SRCS = pulse.php

H5_INC = -I/usr/include/hdf5/serial
UTIL_INC = -I../util

ifeq ($(DEBUG),1)
CXXFLAGS =  -g
else
CXXFLAGS =  -O4
endif

CXX = g++ $(CXXFLAGS) $(H5_INC) $(UTIL_INC)

HDF5_LIB = -lhdf5_serial

.cpp.o:
	$(CXX) -c -o $*.o $<

pulse: pulse.cpp ../util/ph5.o
	$(CXX) -o pulse pulse.cpp ../util/ph5.o $(HDF5_LIB)

write_images: write_images.cpp ../util/ph5.o
	$(CXX) -o write_images write_images.cpp ../util/ph5.o $(HDF5_LIB)

-include Makefile.deps

Makefile.deps: $(SRCS)
	$(CXX) -MM *.cpp > Makefile.deps

clean:
	rm -f $(PROGS) *.o
