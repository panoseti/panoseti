DEBUG = 0

PROGS = image_test img_pulse_test pff_test
OLD_PROGS = ph5_test fits_test pfits_test
    # these require cfitsio and/or hdf5 libraries

OBJS = image.o

all: $(PROGS)

SRCS = image_test.cpp img_pulse_test.cpp
OLD_SRCS = image_test.cpp img_pulse_test.cpp ph5_test.cpp pfits_test.cpp

H5_INC = -I/usr/include/hdf5/serial

ifeq ($(DEBUG),1)
CXXFLAGS =  -g -fPIC
else
CXXFLAGS =  -O4 -fPIC
endif

CXX = g++ $(CXXFLAGS) $(H5_INC)

HDF5_LIB = -lhdf5_serial
FITS_LIB = -lcfitsio

.cpp.o:
	$(CXX) -c -o $*.o $<

image_test: image_test.cpp image.cpp image.h $(OBJS)
	$(CXX) -o image_test image_test.cpp image.cpp

img_pulse_test: img_pulse_test.cpp img_pulse.h
	$(CXX) -o img_pulse_test img_pulse_test.cpp

ph5_test: ph5_test.cpp ph5.o ph5.h
	$(CXX) -o ph5_test ph5_test.cpp ph5.o $(HDF5_LIB)

fits_test: fits_test.cpp
	$(CXX) -o fits_test fits_test.cpp $(FITS_LIB)

pfits_test: pfits_test.cpp pfits.h
	$(CXX) -o pfits_test pfits_test.cpp $(FITS_LIB)
    
pff_test: pff_test.cpp pff.o
	$(CXX) -o pff_test pff_test.cpp pff.o
    
-include Makefile.deps

Makefile.deps: $(SRCS)
	 $(CXX) -MM $(H5_INC) $(SRCS) > Makefile.deps

clean:
	rm -f $(PROGS) *.o
