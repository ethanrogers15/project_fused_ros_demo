from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # Set bag path for playing
    bag_path = '/project_fused_ros_demo/images/regular_testing_two_people/bag' #NOTE: change the path for different data input
    
    # Start model testing node
    workflow_node = Node(
        package='camera_combiner',
        executable='/project_fused_ros_demo/src/camera_combiner/src/workflow.py',
        name='workflow',
        output='screen',
        remappings=[
            ('/lidar_placeholder', '/flexx2_camera_node/depth_image_rect'),
            ('/thermal_placeholder', '/lepton/image_rect'),
            ('/webcam_placeholder', '/webcam/image_rect'),
            ('/lidar_fused_output_placeholder', '/flexx2_camera_node/fused_detection_depth_image_rect'),
            ('/thermal_fused_output_placeholder', '/lepton/fused_detection_image_rect'),
            ('/webcam_fused_output_placeholder', '/webcam/fused_detection_image_rect'),
            ('/lidar_original_output_placeholder', '/flexx2_camera_node/original_detection_depth_image_rect'),
            ('/thermal_original_output_placeholder', '/lepton/original_detection_image_rect'),
            ('/webcam_original_output_placeholder', '/webcam/original_detection_image_rect')
        ],
        parameters=[{ #NOTE: change the following if needed
            'iou_threshold': 0.4, # range from 0 to 1
            'decision_making_mode': 'thermal', # options are 'all', 'thermal', and 'webcam'
            'max_results': 3, # can be any positive integer
            'score_threshold': 0.5 # range from 0 to 1
        }]
    )
    
    # Start RVIZ2 viewing nodes
    rviz2_original_node = Node(
        package='rviz2',
        executable='rviz2',
        name='original_detections',
        output='screen',
        arguments=['-d', '/project_fused_ros_demo/src/camera_combiner/config/workflow_original.rviz']
    )
    
    rviz2_fused_node = Node(
        package='rviz2',
        executable='rviz2',
        name='fused_detections',
        output='screen',
        arguments=['-d', '/project_fused_ros_demo/src/camera_combiner/config/workflow_fused.rviz']
    )

    # Play the testing bag
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path],
        output='screen')
    
    return LaunchDescription([rviz2_original_node, rviz2_fused_node, workflow_node, bag_play])