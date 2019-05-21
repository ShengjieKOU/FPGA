# cmake needs this line
cmake_minimum_required(VERSION 2.8)
project( adas_cmake  )

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -mcpu=cortex-a53")
find_package(catkin REQUIRED COMPONENTS
	  roscpp std_msgs sensor_msgs cv_bridge  image_transport
	  message_generation  )

catkin_package(  CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge
	  image_transport message_runtime)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
message(${catkin_INCLUDE_DIRS})

# find_package(Eigen3)
# include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ./include )
include_directories(include  ${catkin_INCLUDE_DIRS})
#add_subdirectory( src )
add_executable( main src/main.cc )
add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

file(GLOB_RECURSE DPU_CODES model/*.elf)

target_link_libraries(main ${catkin_LIBRARIES})
target_link_libraries(main ${DPU_CODES} ${OpenCV_LIBS})
target_link_libraries(main pthread hineon n2cube dputils)
