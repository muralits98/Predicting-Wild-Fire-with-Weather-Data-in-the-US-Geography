#include <stdio.h>
#include "util.h"

void initialize(void) {
    sqlite3_open("../data.sqlite", &handle);
    const static char* create_table = 
    "CREATE TABLE IF NOT EXISTS AllFire ("
    "City TEXT NOT NULL,"
    "Year TEXT NOT NULL,"
    "DOY INTEGER NOT NULL,"
    "HM TEXT NOT NULL,"
    "Humidity REAL,"
    "Pressure REAL,"
    "Temperature REAL,"
    "WeatherDescription TEXT,"
    "WindDirection REAL,"
    "WindSpeed REAL,"
    "Fire Integer,"
    "PRIMARY KEY (City, Year, DOY, HM)"
    ");";
    sqlite3_exec(handle, create_table, 0, 0, 0);
}

static int handle_insert(void* args, int ncol, char** data, char** fields) {
    static const char *template =
    "INSERT OR IGNORE INTO AllFire VALUES "
    "('%s', '%s', %s, '%s', %s, %s, %s, '%s', %s, %s, %s)";
    static char buf[1000];
    snprintf(buf, 1000, template, (char*)args, data[0],
             data[1], data[2], data[3], data[4], data[5], data[6],
             data[7], data[8], data[9]);
    puts(buf);
    int code = sqlite3_exec(handle, buf, 0, 0, 0);
    printf("errcode %d\n", code);
    return code;
}

int handle_city(void* args, int ncol, char** data, char** fields) {
    static char buf[1000];
    snprintf(buf, 1000, "SELECT * FROM %sFire", data[0]);
    printf("%s\n", buf);
    int code = sqlite3_exec(handle, buf, handle_insert, data[0], 0);
    printf("errcode %d\n", code);
    return code;
}

void finalize(void) {
    sqlite3_close(handle);
}
