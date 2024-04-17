# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -g -G -I./Common

# Source files
SRC := matrixMulMine.cu

# Output executable
OUT := matrixMulMine

# Make all target
all: $(OUT)

# Target for building the executable
$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(OUT)
