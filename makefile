# You may need to edit this file to reflect the type and capabilities of your system.
# The defaults are for a Linux system and may need to be changed for other systems (eg. Mac OS X).


CXX=g++ -std=c++11 -Wall
CC = gcc

#CXX=CC
## When using the Sun Studio compiler


# flags configured by CMake
ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate
else
  LIB_FLAGS = -larmadillo
  ## NOTE: on Ubuntu and Debian based systems you may need to add -lgfortran
  
  #LIB_FLAGS = -larmadillo -library=sunperf
  ## When using the Sun Studio compiler
endif


#DEBUG = -DDEBUG
#GDB = -g


OPT = -O2
## As the Armadillo library uses recursive templates, compilation times depend on the level of optimisation:
##
## -O0: quick compilation, but the resulting program will be slow
## -O1: good trade-off between compilation time and execution speed
## -O2: produces programs which have almost all possible speedups, but compilation takes longer
## -O3: enables auto vectorisation when using gcc

#OPT = -xO4 -xannotate=no
## When using the Sun Studio compiler


#EXTRA_OPT = -fwhole-program
## Uncomment the above line if you're compiling all source files into one program in a single hit


#DEBUG = -DARMA_EXTRA_DEBUG
## Uncomment the above line to enable low-level debugging.
## Lots of debugging information will be printed when a compiled program is run.
## Please enable this option when reporting bugs.


#FINAL = -DARMA_NO_DEBUG
## Uncomment the above line to disable Armadillo's checks.
## Not recommended unless your code has been first thoroughly tested!


CXXFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT)
BIN=./bin/
SOURCE=./source/
NEURAL=neuralnetwork/
IMAGE=ppmreadwriter/

all: imagetest ann

selectiontrainer.o: $(SOURCE)$(NEURAL)selectiontrainer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

selectiontrainer: selectiontrainer.o
	$(CXX) $(CXXFLAGS) $(BIN)$<  -o $@ $(LIB_FLAGS) -lpthread -lm


pgmreader.o: $(SOURCE)$(IMAGE)pgmreader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@ 

annpgm.o: $(SOURCE)$(IMAGE)annpgm.cpp $(SOURCE)$(IMAGE)pgmreader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@ 

ann.o: $(SOURCE)$(NEURAL)ann.cpp $(SOURCE)$(IMAGE)pgmreader.cpp $(SOURCE)$(IMAGE)annpgm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@  

nnanalyzer.o: $(SOURCE)$(NEURAL)nnanalyzer.cpp $(SOURCE)$(NEURAL)ann.cpp $(SOURCE)$(NEURAL)selectiontrainer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

nnmap.o: $(SOURCE)$(NEURAL)nnmap.cpp $(SOURCE)$(NEURAL)ann.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

imagetest.o: $(SOURCE)imagetest.cpp $(SOURCE)$(IMAGE)pgmreader.cpp $(SOURCE)$(NEURAL)ann.cpp $(SOURCE)$(NEURAL)nnanalyzer.cpp $(SOURCE)$(NEURAL)nnmap.cpp $(SOURCE)$(IMAGE)annpgm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

imagetest: imagetest.o pgmreader.o ann.o nnanalyzer.o nnmap.o annpgm.o selectiontrainer.o
	$(CXX) $(CXXFLAGS)  $(BIN)$< $(BIN)pgmreader.o $(BIN)ann.o $(BIN)nnanalyzer.o $(BIN)nnmap.o $(BIN)annpgm.o $(BIN)selectiontrainer.o -o $@ $(LIB_FLAGS) -lpthread -lm



.PHONY: clean

clean:
	rm -f imagetest bin/*.o

