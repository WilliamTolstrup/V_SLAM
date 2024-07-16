cmake_minimum_required(VERSION 3.8)
project(real-time)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(camera src/camera.cpp)

# Link against the required dependencies
ament_target_dependencies(camera rclcpp std_msgs sensor_msgs cv_bridge image_transport)

# Include OpenCV directories
target_include_directories(camera PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(camera ${OpenCV_LIBRARIES})

install(TARGETS
   camera
   DESTINATION lib/${PROJECT_NAME}
 )

ament_package()
