# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qfgao/SLAM/chap07/feature_Extraction

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qfgao/SLAM/chap07/feature_Extraction/build

# Include any dependencies generated for this target.
include CMakeFiles/feature_Extraction.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/feature_Extraction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/feature_Extraction.dir/flags.make

CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o: CMakeFiles/feature_Extraction.dir/flags.make
CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o: ../feature_extraction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qfgao/SLAM/chap07/feature_Extraction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o -c /home/qfgao/SLAM/chap07/feature_Extraction/feature_extraction.cpp

CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qfgao/SLAM/chap07/feature_Extraction/feature_extraction.cpp > CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.i

CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qfgao/SLAM/chap07/feature_Extraction/feature_extraction.cpp -o CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.s

CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.requires:

.PHONY : CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.requires

CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.provides: CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.requires
	$(MAKE) -f CMakeFiles/feature_Extraction.dir/build.make CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.provides.build
.PHONY : CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.provides

CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.provides.build: CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o


# Object files for target feature_Extraction
feature_Extraction_OBJECTS = \
"CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o"

# External object files for target feature_Extraction
feature_Extraction_EXTERNAL_OBJECTS =

feature_Extraction: CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o
feature_Extraction: CMakeFiles/feature_Extraction.dir/build.make
feature_Extraction: /usr/local/lib/libopencv_gapi.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_objdetect.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_ml.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_photo.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_video.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_highgui.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_stitching.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_dnn.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_videoio.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_calib3d.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_features2d.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_flann.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_imgproc.so.4.2.0
feature_Extraction: /usr/local/lib/libopencv_core.so.4.2.0
feature_Extraction: CMakeFiles/feature_Extraction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qfgao/SLAM/chap07/feature_Extraction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable feature_Extraction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/feature_Extraction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/feature_Extraction.dir/build: feature_Extraction

.PHONY : CMakeFiles/feature_Extraction.dir/build

CMakeFiles/feature_Extraction.dir/requires: CMakeFiles/feature_Extraction.dir/feature_extraction.cpp.o.requires

.PHONY : CMakeFiles/feature_Extraction.dir/requires

CMakeFiles/feature_Extraction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/feature_Extraction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/feature_Extraction.dir/clean

CMakeFiles/feature_Extraction.dir/depend:
	cd /home/qfgao/SLAM/chap07/feature_Extraction/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qfgao/SLAM/chap07/feature_Extraction /home/qfgao/SLAM/chap07/feature_Extraction /home/qfgao/SLAM/chap07/feature_Extraction/build /home/qfgao/SLAM/chap07/feature_Extraction/build /home/qfgao/SLAM/chap07/feature_Extraction/build/CMakeFiles/feature_Extraction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/feature_Extraction.dir/depend

