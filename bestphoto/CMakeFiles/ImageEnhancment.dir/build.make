# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto

# Include any dependencies generated for this target.
include CMakeFiles/ImageEnhancment.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ImageEnhancment.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ImageEnhancment.dir/flags.make

CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o: CMakeFiles/ImageEnhancment.dir/flags.make
CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o: ImageEnhancment.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o -c /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto/ImageEnhancment.cpp

CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto/ImageEnhancment.cpp > CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.i

CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto/ImageEnhancment.cpp -o CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.s

CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.requires:
.PHONY : CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.requires

CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.provides: CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.requires
	$(MAKE) -f CMakeFiles/ImageEnhancment.dir/build.make CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.provides.build
.PHONY : CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.provides

CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.provides.build: CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o

# Object files for target ImageEnhancment
ImageEnhancment_OBJECTS = \
"CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o"

# External object files for target ImageEnhancment
ImageEnhancment_EXTERNAL_OBJECTS =

ImageEnhancment: CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o
ImageEnhancment: CMakeFiles/ImageEnhancment.dir/build.make
ImageEnhancment: /usr/local/lib/libopencv_core.a
ImageEnhancment: /usr/local/lib/libopencv_flann.a
ImageEnhancment: /usr/local/lib/libopencv_imgproc.a
ImageEnhancment: /usr/local/lib/libopencv_highgui.a
ImageEnhancment: /usr/local/lib/libopencv_features2d.a
ImageEnhancment: /usr/local/lib/libopencv_calib3d.a
ImageEnhancment: /usr/local/lib/libopencv_ml.a
ImageEnhancment: /usr/local/lib/libopencv_video.a
ImageEnhancment: /usr/local/lib/libopencv_legacy.a
ImageEnhancment: /usr/local/lib/libopencv_objdetect.a
ImageEnhancment: /usr/local/lib/libopencv_photo.a
ImageEnhancment: /usr/local/lib/libopencv_gpu.a
ImageEnhancment: /usr/local/lib/libopencv_videostab.a
ImageEnhancment: /usr/local/lib/libopencv_ts.a
ImageEnhancment: /usr/local/lib/libopencv_ocl.a
ImageEnhancment: /usr/local/lib/libopencv_superres.a
ImageEnhancment: /usr/local/lib/libopencv_nonfree.a
ImageEnhancment: /usr/local/lib/libopencv_stitching.a
ImageEnhancment: /usr/local/lib/libopencv_contrib.a
ImageEnhancment: /usr/local/lib/libopencv_nonfree.a
ImageEnhancment: /usr/local/lib/libopencv_gpu.a
ImageEnhancment: /usr/local/lib/libopencv_legacy.a
ImageEnhancment: /usr/local/lib/libopencv_photo.a
ImageEnhancment: /usr/local/lib/libopencv_ocl.a
ImageEnhancment: /usr/local/lib/libopencv_calib3d.a
ImageEnhancment: /usr/local/lib/libopencv_features2d.a
ImageEnhancment: /usr/local/lib/libopencv_flann.a
ImageEnhancment: /usr/local/lib/libopencv_ml.a
ImageEnhancment: /usr/local/lib/libopencv_video.a
ImageEnhancment: /usr/local/lib/libopencv_objdetect.a
ImageEnhancment: /usr/local/lib/libopencv_highgui.a
ImageEnhancment: /usr/local/lib/libopencv_imgproc.a
ImageEnhancment: /usr/local/lib/libopencv_core.a
ImageEnhancment: /usr/local/share/OpenCV/3rdparty/lib/liblibjpeg.a
ImageEnhancment: /usr/local/share/OpenCV/3rdparty/lib/liblibpng.a
ImageEnhancment: /usr/local/share/OpenCV/3rdparty/lib/liblibtiff.a
ImageEnhancment: /usr/local/share/OpenCV/3rdparty/lib/liblibjasper.a
ImageEnhancment: /usr/local/share/OpenCV/3rdparty/lib/libIlmImf.a
ImageEnhancment: /usr/local/share/OpenCV/3rdparty/lib/libzlib.a
ImageEnhancment: CMakeFiles/ImageEnhancment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ImageEnhancment"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ImageEnhancment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ImageEnhancment.dir/build: ImageEnhancment
.PHONY : CMakeFiles/ImageEnhancment.dir/build

CMakeFiles/ImageEnhancment.dir/requires: CMakeFiles/ImageEnhancment.dir/ImageEnhancment.cpp.o.requires
.PHONY : CMakeFiles/ImageEnhancment.dir/requires

CMakeFiles/ImageEnhancment.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ImageEnhancment.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ImageEnhancment.dir/clean

CMakeFiles/ImageEnhancment.dir/depend:
	cd /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto /Users/sami/Desktop/Semester8/ELE882/Brizi/bestphoto/CMakeFiles/ImageEnhancment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ImageEnhancment.dir/depend

