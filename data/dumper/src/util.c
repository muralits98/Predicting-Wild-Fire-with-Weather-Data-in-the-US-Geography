#include <stdio.h>
#include "util.h"

static _Bool header = 0;

void initialize(void) {
    sqlite3_open("../data.sqlite", &handle);
}

static int handle_dump(void* args, int ncol, char** data, char** fields) {
    if (header) {
        fprintf((FILE*)args, "%s", fields[0]);
        for (int i = 1; i < ncol; i++) {
            fprintf((FILE*)args, ",%s", fields[i]);
        }
        fputc(10, (FILE*)args);
        header = 0;
    }
    fprintf((FILE*)args, "%s", data[0]);
    for (int i = 1; i < ncol; i++) {
        fprintf((FILE*)args, ",%s", data[i]);
    }
    fputc(10, (FILE*)args);
    return 0;
}

int handle_table_name(void* args, int ncol, char** data, char** fields) {
    char buf[1000];
    snprintf(buf, 1000, "../db_dump/%s.csv", data[0]);
    FILE* fout = fopen(buf, "w");
    header = 1;
    snprintf(buf, 1000, "SELECT * FROM %s", data[0]);
    printf("%s errcode ", buf);
    int code = sqlite3_exec(handle, buf, handle_dump, fout, 0);
    printf("%d\n", code);
    header = 0;
    fclose(fout);
    return code;
}

void finalize(void) {
    sqlite3_close(handle);
}
