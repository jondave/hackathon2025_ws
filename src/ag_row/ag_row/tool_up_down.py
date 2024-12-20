import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry  # Updated import for Odometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class PositionControllerPublisher(Node):
    def __init__(self):
        super().__init__('position_controller_publisher')
        
        # Create a publisher for the topic
        self.publisher_ = self.create_publisher(Float64MultiArray, '/robot/implement/position_controller/commands', 10)
        
        # Create a subscriber for the filtered_odom topic (Odometry message)
        self.subscription = self.create_subscription(
            Odometry,  # Updated message type to Odometry
            '/robot/localisation/filtered_odom',
            self.odom_callback,
            10
        )
        
        # Timer to periodically call the publish function
        self.timer = self.create_timer(1.0, self.publish_position_data)
        
        # Variable to store the current pose position (x, y)
        self.pose_position = [0.0, 0.0]

        # Define the coordinates for mixed field and sloping field as Polygons
        self.coords_mixed_field = [
            [98.734000, 99.966800],
            [93.379110, 111.679380],
            [125.239622, 125.099028],
            [129.062393, 111.765516]
        ]
        
        self.coords_sloping_field = [
            [-14.944984, 187.037058],
            [41.963383, 212.281830],
            [-27.886276, 197.237122],
            [34.530182, 222.108971]
        ]
        
        # Convert the coordinates to Shapely Polygons
        self.mixed_field_polygon = Polygon(self.coords_mixed_field)
        self.sloping_field_polygon = Polygon(self.coords_sloping_field)

    def odom_callback(self, msg):
        """Callback function that is called when a message is received on the filtered_odom topic."""
        # Extract the position (x, y) from the incoming Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.pose_position = [x, y]
        self.get_logger().info(f'Received position: x = {x}, y = {y}')

    def publish_position_data(self):
        """Publish a Float64MultiArray message based on position."""
        # Create a Float64MultiArray message
        msg = Float64MultiArray()
        
        # Create a Shapely Point object from the current position
        current_position = Point(self.pose_position[0], self.pose_position[1])

        # Log the current position
        # self.get_logger().info(f'x: {self.pose_position[0]}')
        # self.get_logger().info(f'y: {self.pose_position[1]}')
        # self.get_logger().info(f'current_position: {current_position}')

        # Check if the current position is inside either of the polygons (mixed field or sloping field)
        if self.mixed_field_polygon.contains(current_position) or self.sloping_field_polygon.contains(current_position):
            data = [1.0]  # Inside the defined areas, publish 1.0
        else:
            data = [0.0]  # Outside the defined areas, publish 0.0
        
        # Assign the data to the message
        msg.data = data

        # Publish the message to the position_controller/commands topic
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {data}')

def main(args=None):
    rclpy.init(args=args)

    # Create an instance of the publisher-subscriber node
    node = PositionControllerPublisher()

    try:
        # Spin the node so it continues to process subscriptions and publishing
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node when done
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
