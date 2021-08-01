all: pulse_test ph5_test fits_test

SRCS = pulse_test.cpp ph5_test.cpp

H5_INC = -I/usr/include/hdf5/serial

#CXX =  g++ -O4 -I$(H5_INC)
CXX =  g++ -g $(H5_INC)

HDF5_LIB = -lhdf5_serial

FITS_INC = -I/usr/local/new_cfitsio/include
FITS_LIB = -L/usr/local/new_cfitsio/lib -lcfitsio

.cpp.o:
	$(CXX) -c -o $*.o $<

pulse_test: pulse_test.cpp pulse_find.h
	$(CXX) -o pulse_test pulse_test.cpp

ph5_test: ph5_test.cpp ph5.o ph5.h
	$(CXX) -o ph5_test ph5_test.cpp ph5.o $(HDF5_LIB)

fits_test: fits_test.cpp
	$(CXX) $(FITS_INC) -o fits_test fits_test.cpp $(FITS_LIB)
    
-include Makefile.deps

Makefile.deps: $(SRCS)
	 $(CXX) -MM *.cpp > Makefile.deps
