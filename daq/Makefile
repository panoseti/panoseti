CC          = g++
HDF5_LIB_CCFLAGS = -I/usr/local/hdf5/include \
					 -L/usr/local/hdf5/lib \
					 -lhdf5_hl \
					 -lhdf5
REDIS_LIB_CCFLAGS = -lhiredis
HSD_LIB_CCFLAGS     = -g -O3 -fPIC -shared -lstdc++ -mavx -msse4 \
                     -I. -I$(CUDA_DIR)/include -I/usr/local/include \
                     -L. -L/usr/local/lib \
                     -lhashpipe -lrt -lm \
					 $(HDF5_LIB_CCFLAGS) \
					 $(REDIS_LIB_CCFLAGS) \
					 -lz -ldl -lm \
					 -Wl,-rpath
HSD_LIB_TARGET   = HSD_hashpipe.so
HSD_LIB_SOURCES  = HSD_net_thread.c \
		      HSD_compute_thread.c \
		      HSD_output_thread.c \
                      HSD_databuf.c
HSD_LIB_INCLUDES = HSD_databuf.h

all: $(HSD_LIB_TARGET)

$(HSD_LIB_TARGET): $(HSD_LIB_SOURCES) $(HSD_LIB_INCLUDES)
	$(CC) -o $(HSD_LIB_TARGET) $(HSD_LIB_SOURCES) $(HSD_LIB_CCFLAGS)

tags:
	ctags -R .
clean:
	rm -f $(HSD_LIB_TARGET) tags

prefix=/usr/local
LIBDIR=$(prefix)/lib
BINDIR=$(prefix)/bin
install-lib: $(HSD_LIB_TARGET)
	mkdir -p "$(DESTDIR)$(LIBDIR)"
	install -p $^ "$(DESTDIR)$(LIBDIR)"
	rm $^
install: install-lib

.PHONY: all tags clean install install-lib
# vi: set ts=8 noet :
