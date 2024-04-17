# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -g -G -I../../cuda-samples/Common

# Source files
SRC := dft.cu

# Output executable
OUT := dft

# Make all target
all: $(OUT)

# Target for building the executable
$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(OUT)
