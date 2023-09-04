include config.mk

all: lib/libselectivex.a

static: lib/libselectivex.a

shared: lib/libselectivex.so

test: lib/libselectivex.a

lib/libselectivex.a:\
		obj/util.o\
		obj/interface.o\
		obj/model.o\
		obj/intercept_comp.o\
		obj/intercept_comm.o
	ar -crs lib/libselectivex.a\
					obj/util.o\
					obj/interface.o\
					obj/model.o\
					obj/intercept_comp.o\
					obj/intercept_comm.o

lib/libselectivex.so:\
		obj/util.o\
		obj/interface.o\
		obj/model.o\
		obj/intercept_comp.o\
		obj/intercept_comm.o
	$(CXX) -shared -o lib/libselectivex.so\
					obj/util.o\
					obj/interface.o\
					obj/model.o\
					obj/intercept_comp.o\
					obj/intercept_comm.o

obj/util.o: src/util.cxx
	$(CXX) src/util.cxx -c -o obj/util.o $(CXXFLAGS)

obj/interface.o: src/interface.cxx
	$(CXX) src/interface.cxx -c -o obj/interface.o $(CXXFLAGS)

obj/model.o: src/model.cxx
	$(CXX) src/model.cxx -c -o obj/model.o $(CXXFLAGS)

obj/intercept_comp.o: src/intercept/comp.cxx
	$(CXX) src/intercept/comp.cxx -c -o obj/intercept_comp.o $(CXXFLAGS)

obj/intercept_comm.o: src/intercept/comm.cxx
	$(CXX) src/intercept/comm.cxx -c -o obj/intercept_comm.o $(CXXFLAGS)

clean:
	rm -f obj/*.o lib/libselectivex.a lib/libselectivex.so
