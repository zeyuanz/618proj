
CXXFLAGS += -std=c++11 -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG

.phony: all 

all: 
	g++ wsp_SA_space.cpp -o wsp_SA_space $(CXXFLAGS)
	g++ wsp_SA_double.cpp -o wsp_SA_double $(CXXFLAGS)
clean:
	rm -f ./wsp_SA_double
	rm -f ./wsp_SA_space
