# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jz853/cmake-3.18.5-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/jz853/cmake-3.18.5-Linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release

# Include any dependencies generated for this target.
include CMakeFiles/Driver.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Driver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Driver.dir/flags.make

CMakeFiles/Driver.dir/Driver.cpp.o: CMakeFiles/Driver.dir/flags.make
CMakeFiles/Driver.dir/Driver.cpp.o: ../../Driver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Driver.dir/Driver.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Driver.dir/Driver.cpp.o -c /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/Driver.cpp

CMakeFiles/Driver.dir/Driver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Driver.dir/Driver.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/Driver.cpp > CMakeFiles/Driver.dir/Driver.cpp.i

CMakeFiles/Driver.dir/Driver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Driver.dir/Driver.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/Driver.cpp -o CMakeFiles/Driver.dir/Driver.cpp.s

# Object files for target Driver
Driver_OBJECTS = \
"CMakeFiles/Driver.dir/Driver.cpp.o"

# External object files for target Driver
Driver_EXTERNAL_OBJECTS =

Driver: CMakeFiles/Driver.dir/Driver.cpp.o
Driver: CMakeFiles/Driver.dir/build.make
Driver: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
Driver: /usr/lib/x86_64-linux-gnu/libpthread.so
Driver: CMakeFiles/Driver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Driver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Driver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Driver.dir/build: Driver

.PHONY : CMakeFiles/Driver.dir/build

CMakeFiles/Driver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Driver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Driver.dir/clean

CMakeFiles/Driver.dir/depend:
	cd /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2 /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2 /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release /home/jz853/Dropbox/Interview/cytochip/Algorithm_questionnaire/sols/Q2/build/release/CMakeFiles/Driver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Driver.dir/depend
