CC=g++
CFLAGS_OMP=-fopenmp -g
LDFLAGS_OMP=-fopenmp
CFLAGS_SIMD=-Wall -sse2
ERROR_CHECKING=-Wall -Wextra -Werror

img_reduce: img_reduce.o
	$(CC) $(LDFLAGS_OMP) $^ -o $@

img_reduce.o: img_reduce.cpp
	$(CC) $(CFLAGS_OMP) $< -o $@ $(ERROR_CHECKING)

clean:
	rm -f img_reduce
	rm -f *.o