cmake_minimum_required(VERSION 3.5)
project(camera_combiner)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rviz2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(usb_cam REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pluginlib REQUIRED)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# # Install config files
# install(DIRECTORY config/
#   DESTINATION share/${PROJECT_NAME}/config
# )

# Install Python scripts
install(PROGRAMS
  src/workflow.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
