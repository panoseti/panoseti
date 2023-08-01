CC=gcc
CFLAGS_OMP=-std=gnu99 -fopenmp -c -g
LDFLAGS_OMP=-fopenmp

img_reduce: img_reduce.o
	$(CC) $(LDFLAGS_OMP) $^ -o $@

img_reduce.o: img_reduce.c
	$(CC) $(CFLAGS_OMP) $< -o $@

clean:
	rm -f img_reduce
	rm -f *.o