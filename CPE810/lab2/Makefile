# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS :=  -I../../cuda-samples/Common

# Source files
SRC := histogram.cu

# Output executable
OUT := hist

# Make all target
all: $(OUT)

# Target for building the executable
$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(OUT)
