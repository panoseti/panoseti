all: pulse_test ph5_test

H5_INC = /usr/include/hdf5/serial

#CXX =  g++ -O4 -I$(H5_INC)
CXX =  g++ -g -I$(H5_INC)

HDF5_LIB = -lhdf5_serial

.cpp.o:
	$(CXX) -c -o $*.o $<

pulse_test: pulse_test.cpp pulse_find.h
	$(CXX) -o pulse_test pulse_test.cpp

ph5_test: ph5_test.cpp ph5.o ph5.h
	$(CXX) -o ph5_test ph5_test.cpp ph5.o $(HDF5_LIB)
