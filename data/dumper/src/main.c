#include "util.h"

const char* table_name = "SELECT name FROM sqlite_master "
                         "WHERE type = 'table' AND "
                         "name NOT LIKE '%%\\_%' ESCAPE '\\' AND "
                         "name <> 'KNN' AND name NOT LIKE 'Spatial%' AND "
                         "name NOT LIKE 'Elementary%%'";

int main(void) {
    initialize();
    sqlite3_exec(handle, table_name, handle_table_name, 0, 0);
    finalize();
}
