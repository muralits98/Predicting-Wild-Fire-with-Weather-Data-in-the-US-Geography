#include <stdio.h>
#include <stdlib.h>
#include "util.h"

void initialize(void) {
    sqlite3_open("../data.sqlite", &handle);
}

const char* city = "SELECT * FROM Fires WHERE "
                   "FIRE_YEAR >= 2012 AND FIRE_YEAR <= 2017 AND "
                   "DISCOVERY_DOY IS NOT NULL AND "
                   "DISCOVERY_TIME IS NOT NULL AND "
                   "CONT_DOY IS NOT NULL AND "
                   "CONT_TIME IS NOT NULL AND "
                   "LATITUDE >= %lf AND LATITUDE <= %lf AND "
                   "LONGITUDE >= %lf AND LONGITUDE <= %lf";

const char* update = "UPDATE %sFire "
                     "SET Fire = %d WHERE "
                     "Year = %s AND "
                     "DOY >= %s AND DOY <= %s AND "
                     "HM >= %s AND HM <= %s";

static int handle_fire(void* args, int ncol, char** data, char** fields) {
    static char buf[1000];
    snprintf(buf, 1000, update, (char*)args, 1, data[19], data[21], data[26],
             data[22], data[27]);
    puts(buf);
    int code = sqlite3_exec(handle, buf, 0, 0, 0);
    printf("error code %d\n", code);
    return code;
}

int handle_city(void* args, int ncol, char** data, char** fields) {
    static char buf[1000];
    for (int i = 0; i < ncol; i++) {
        printf("%s ", data[i]);
    }
    putchar(10);
    double lat = atof(data[1]);
    double lon = atof(data[2]);
    snprintf(buf, 1000, city, lat - 1, lat + 1, lon - 1, lon + 1);
    puts(buf);
    int code = sqlite3_exec(handle, buf, handle_fire, data[0], 0);
    printf("error code %d\n", code);
    return code;
}

void finalize(void) {
    sqlite3_close(handle);
}
