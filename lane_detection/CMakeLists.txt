cmake_minimum_required(VERSION 2.8.3)
project(lane_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  message_generation 
)
 add_message_files(
    FILES
    ld_LaneParam.msg
    ld_Frame.msg
    ld_Point.msg
    ld_Coeff.msg
 )
 generate_messages(
   DEPENDENCIES
   std_msgs
 )
catkin_package(
#INCLUDE_DIRS include
#  LIBRARIES idriverSensorGps
   CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge
  image_transport
#  DEPENDS system_lib
   message_runtime)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -msse3 -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g  -std=c++11")
find_package( OpenCV REQUIRED )
message(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})
#add_library(moo src/LaneDetection.cpp)
add_executable(lane_detection src/IPM.cpp src/main_ros.cpp src/LaneDetection.cpp   src/JudgeLane.cpp)  
#target_link_libraries(lane_detection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} stdc++)
target_link_libraries(lane_detection ${catkin_LIBRARIES} ${OpenCV_LIBS} )
add_dependencies(lane_detection lane_detection_generate_messages_cpp)


