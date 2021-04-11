
CXXFLAGS += -std=c++11 -Wall -Wextra -g -O3 -DNDEBUG

.phony: all 

all: 
	g++ wsp_SA_serial.cpp -o wsp_SA_serial $(CXXFLAGS)

clean:
	rm -f ./wsp_SA_serial
