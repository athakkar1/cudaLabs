# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS :=  -I../../cuda-samples/Common

# Source files
SRC := convolution.cu

# Output executable
OUT := conv

# Make all target
all: $(OUT)

# Target for building the executable
$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(OUT)
