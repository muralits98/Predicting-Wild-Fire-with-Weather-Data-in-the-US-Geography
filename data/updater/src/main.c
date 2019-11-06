#include "util.h"

int main(void) {
    initialize();
    sqlite3_exec(handle, "SELECT * FROM Cities", handle_city, 0, 0);
    finalize();
}
