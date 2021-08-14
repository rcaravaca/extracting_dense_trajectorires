# set the binaries that have to be built
TARGETS := DenseTrack Video

# set the build configuration set 
BUILD := release
#BUILD := debug

# set bin and build dirs
BUILDDIR := .build_$(BUILD)
BINDIR := $(BUILD)

# libraries 
LDLIBS = $(addprefix -l, $(LIBS) $(LIBS_$(notdir $*)))
LIBS := \
	opencv_videoio opencv_core opencv_highgui opencv_video opencv_imgproc opencv_imgcodecs \
	avformat avdevice avutil avcodec swscale

# set some flags and compiler/linker specific commands
CXXFLAGS = -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) -I. -I/opt/include -I/usr/include/python3.9 -I/usr/include/x86_64-linux-gnu/python3.9 -lpython3.9 -lcrypt -lpthread -ldl  -lutil -lm -lm 
CXXFLAGS_debug := -ggdb
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
LDFLAGS = -I/usr/include/python3.9 -L/usr/lib -pipe -Wall $(LDFLAGS_$(BUILD)) -lpython3.9 -lcrypt -lpthread -ldl  -lutil -lm -lm 
LDFLAGS_debug := -ggdb
LDFLAGS_release := -O3 -ggdb

include make/generic.mk
