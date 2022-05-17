DEBUG = 1

PROGS = pulse write_images

all: $(PROGS)

SRCS = pulse.php

UTIL_INC = -I../util

ifeq ($(DEBUG),1)
CXXFLAGS =  -g
else
CXXFLAGS =  -O4
endif

CXX = g++ $(CXXFLAGS) $(H5_INC) $(UTIL_INC)

.cpp.o:
	$(CXX) -c -o $*.o $<

pulse: pulse.cpp ../util/ph5.o ../util/pff.o
	$(CXX) -o pulse pulse.cpp ../util/pff.o

write_images: write_images.cpp ../util/pff.o
	$(CXX) -o write_images write_images.cpp ../util/pff.o

-include Makefile.deps

Makefile.deps: $(SRCS)
	$(CXX) -MM *.cpp > Makefile.deps

clean:
	rm -f $(PROGS) *.o
