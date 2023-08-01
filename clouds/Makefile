CC=gcc
CFLAGS_OMP=-fopenmp -c -g
LDFLAGS_OMP=-fopenmp
CFLAGS_SIMD=-Wall -sse2

img_reduce: img_reduce.o
	$(CC) $(LDFLAGS_OMP) $^ -o $@

img_reduce.o: img_reduce.cpp
	$(CC) $(CFLAGS_OMP) $< -o $@

clean:
	rm -f img_reduce
	rm -f *.o
