#ifndef _UTIL_H_
#define _UTIL_H
#include <sqlite3.h>

sqlite3* handle;

void initialize(void);
int handle_city(void* args, int ncol, char** data, char** fields);
void finalize(void);
#endif
