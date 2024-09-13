# Compiler
CXX = clang++
# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# Include directories
INCLUDES = -I./OMPEval

# Library directories
LIBDIRS = -L./OMPEval/lib

# Libraries to link
LIBS = -lompeval

# Source files
SRCS = $(wildcard src/*.cpp)

# Object files
OBJS = $(SRCS:.cpp=.o)

# Output executable
TARGET = main

# Default target
all: $(TARGET)

# Rule to link the program
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBDIRS) $^ -o $@ $(LIBS)

# Rule to compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build the OMPEval library
ompeval_lib:
	$(MAKE) -C OMPEval

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)
	$(MAKE) -C OMPEval clean

# Run the program
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean ompeval_lib run