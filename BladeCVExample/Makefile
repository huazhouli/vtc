topdir=/usr/local/AID/bladecv/
GTK-VERSION = gtk+-3.0

incdir = $(topdir)/include
libdir = $(topdir)/libs

linksyslibs = -lpthread -ldl -lm `pkg-config --libs $(GTK-VERSION)`

OLEVEL      = -O3 -Wdeprecated-declarations

CFLAGS   += -Wall $(OLEVEL) $(PLATFORM) 


CFLAGS += -DPERF_STATISTICS
CFLAGS   += -I$(incdir)
CFLAGS += `pkg-config --cflags $(GTK-VERSION)`

CXXFLAGS += -Wall $(OLEVEL) $(PLATFORM)
CXXFLAGS += -DFASTCV_DEBUG
CXXFLAGS += -DPERF_STATISTICS

CXXFLAGS += `pkg-config --cflags $(GTK-VERSION)`
LDFLAGS = -I$(topdir)/include -L$(topdir)/libs -lbcv -I$(topdir)/lib_core/include -I$(topdir)/lib_imgcodecs/include -I$(topdir)/lib_imgproc/include -I$(topdir)/libbcv_op/include  -lfmat_core -I/usr/include/cairo/ -I/usr/include/ -lrockchip_mpp -lrockchip_rga
EXES = test
all : $(EXES)
test : test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
clean :
	$(RM) $(EXES)
