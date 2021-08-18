#include <stdio.h>
#include <string.h>

int main(int, char**) {
    short img[1024];
    const char* j = "{\"foo\": 1.0}\n";
    int n = strlen(j);

    FILE* f = fopen("raw.dat", "w");
    for (int i=0; i<10000000; i++) {
        fwrite(j, n, 1, f);
        fwrite(img, 1024, sizeof(short), f);
    }
    fclose(f);
}
