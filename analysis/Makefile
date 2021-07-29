all: pulse

H5_INC = -I/usr/include/hdf5/serial
UTIL_INC = -I../util

#CXX =  g++ -O4 $(H5_INC) $(UTIL_INC)
CXX =  g++ -g $(H5_INC) $(UTIL_INC)

HDF5_LIB = -lhdf5_serial

.cpp.o:
	$(CXX) -c -o $*.o $<

pulse: pulse.cpp
	$(CXX) -o pulse pulse.cpp ../util/ph5.o $(HDF5_LIB)

-include Makefile.deps

Makefile.deps: $(SRCS)
	$(CXX) -MM *.cpp > Makefile.deps
