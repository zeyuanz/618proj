#15-618 Project

##FILES
- Makefile. For compiling. Shoule modify to make it more robust.
- wsp-ref. Correct implementaion of WSP problem solver. Used for comparison.
- wsp.h. Library with general WSP related functions.
- wsp_SA_serial.cpp. Serial implementation of SA for WSP.

##USAGE
compile
`make`
Print program usage
`wsp_SA_serial -h'

##TODO
1. Verify correctness
2. Finish baseline of WSP parallel
3. Finish serial non-convex optimization SA
4. Finish baseline of non-convex optimization parallel SA

