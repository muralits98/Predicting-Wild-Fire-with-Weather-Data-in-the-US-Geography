CC = gcc
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
BIN = dumper 

%.o: %.c
	$(CC) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(OBJ) -o $(BIN) -lsqlite3

.PHONY: clean
clean: $(BIN)
	@rm $(OBJ) $(BIN)
