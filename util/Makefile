DEBUG = 1

PROGS = pulse_test ph5_test fits_test pfits_test pff_test

all: $(PROGS)

SRCS = pulse_test.cpp ph5_test.cpp pfits_test.cpp

H5_INC = -I/usr/include/hdf5/serial

ifeq ($(DEBUG),1)
CXXFLAGS =  -g
else
CXXFLAGS =  -O4
endif

CXX = g++ $(CXXFLAGS) $(H5_INC)

HDF5_LIB = -lhdf5_serial
FITS_LIB = -lcfitsio

.cpp.o:
	$(CXX) -c -o $*.o $<

pulse_test: pulse_test.cpp pulse_find.h
	$(CXX) -o pulse_test pulse_test.cpp

ph5_test: ph5_test.cpp ph5.o ph5.h
	$(CXX) -o ph5_test ph5_test.cpp ph5.o $(HDF5_LIB)

fits_test: fits_test.cpp
	$(CXX) -o fits_test fits_test.cpp $(FITS_LIB)

pfits_test: pfits_test.cpp pfits.h
	$(CXX) -o pfits_test pfits_test.cpp $(FITS_LIB)
    
pff_test: pff_test.cpp
	$(CXX) -o pff_test pff_test.cpp
    
-include Makefile.deps

Makefile.deps: $(SRCS)
	 $(CXX) -MM $(H5_INC) *.cpp > Makefile.deps

clean:
	rm -f $(PROGS) *.o
