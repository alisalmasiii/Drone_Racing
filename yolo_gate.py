import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import numpy as np
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import time

# PID gains
kp = 0.1
ki = 0.01
kd = 0.05
# Initialize the PID controller variables
integral = 0
prev_e = 0
up_flag = True


class YoloSubscriber(Node):

    def __init__(self):
        super().__init__('yolo_subscriber_node')

        # Load YOLO model
        self.model = YOLO('/ARMRS/yolo/best.pt')

        # Image bridge
        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        # Subscribe to image_raw topic
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            qos_profile)

        self.get_logger().info('YOLOv11 inference node started, listening to /image_raw')
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # qos change because returned error with qos profile
        self.get_logger().info("Tello controller initialized")
        




    def listener_callback(self, msg):
        # Convert ROS image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLO inference
        results = self.model(frame, conf = 0.34)
        boxes = results[0].boxes

        target_gate = None
        stop_detected = None
        max_gate_area = 0

        fly = Twist()
        def pid_ctrl(error):
            global integral, prev_e
            integral += error
            der = error - prev_e
            pid_out = kp * error + ki * integral + kd * der
            prev_e = error
            return pid_out
        
        global up_flag
        if up_flag == True:
            print("up flag")
            up_flag = False
            self.fly(0.0, 0.0, 0.7, 0.0)
            time.sleep(1)
            self.fly(0.0, 0.0, 0.0, 0.0)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())  # Get class ID
                class_name = self.model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)

                if class_name == 'gate':
                    if area > max_gate_area:
                        max_gate_area = area
                        target_gate = (x1, y1, x2, y2)

                elif class_name == 'stop':
                    stop_detected = (x1, y1, x2, y2)

        # Prioritize gate detection
        if target_gate:
            x1, y1, x2, y2 = map(int, target_gate)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, 'Target Gate', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            #print("max_gate_area", (max_gate_area)//10000)
            #print (35000.0/(max_gate_area)) 

            centroid_x = (x1 + (x2-x1)//2) 
            centroid_y = ( y1 + (y2-y1)//2 ) + 160
            cv2.circle(frame, (centroid_x, centroid_y), 20, (255, 0, 0), 2)
            error_x = centroid_x - frame.shape[1] // 2  # Center of the frame
            error_y = centroid_y - frame.shape[0] // 2

            # Gate control logic here
            self.get_logger().info("Gate detected — navigating.")
            
            if abs(error_y) > 25:  # if the gate is too high or low
                pid_out = pid_ctrl(error_y) # still to consider how to apply pid output
                if error_y < 25:
                    self.fly(0.0, 0.0, 0.16, 0.0)
                    self.get_logger().info("Flying up")
                else:
                    self.fly(0.0, 0.0, -0.16, 0.0)
                    self.get_logger().info("Flying down")

            elif abs(error_x) > 25:  # if the gate is on the left or right
                pid_out = pid_ctrl(error_x) # still to consider how to apply pid output
                #fly.linear.y = float(np.clip(pid_out, -0.3, 0.3))  # to use pid as a clipped output
                if error_x < 25:
                    self.fly(0.0, 0.0, 0.0, 0.09)
                    self.get_logger().info("Flying left")
                else:
                    self.fly(0.0, 0.0, 0.0, -0.09)
                    self.get_logger().info("Flying right")
            

            else:  # If it's near the center, move forward
                
                self.fly(0.4, 0.0, 0.0, 0.0)
                self.get_logger().info("Flying forward")
                time.sleep(2.00)
                #time.sleep(88000.0/(max_gate_area))  # waits for 10 seconds
                print("Moving on.")

        # Only consider stop sign if no gates are visible
        elif stop_detected:
            x1, y1, x2, y2 = map(int, stop_detected)
            width = x2 - x1
            height = y2 - y1
            #print(width, height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, 'Stop Sign', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            centroid_x = ( x1 + (x2-x1)//2 ) - 20
            centroid_y = ( y1 + (y2-y1)//2 ) + 180
            cv2.circle(frame, (centroid_x, centroid_y), 20, (255, 0, 0), 2)
            error_x = centroid_x - frame.shape[1] // 2  # Center of the frame
            error_y = centroid_y - frame.shape[0] // 2
            
            if abs(error_y) > 45:  # if the gate is too high or low
                pid_out = pid_ctrl(error_y) # still to consider how to apply pid output
                if error_y < 45:
                    self.fly(0.0, 0.0, 0.15, 0.0)
                    self.get_logger().info("Flying up")
                else:
                    self.fly(0.0, 0.0, -0.15, 0.0)
                    print("Flying down")

            elif abs(error_x) > 45:  # if the gate is on the left or right
                pid_out = pid_ctrl(error_x) # still to consider how to apply pid output
                #fly.linear.y = float(np.clip(pid_out, -0.3, 0.3))  # to use pid as a clipped output
                if error_x < 45:
                    self.fly(0.0, 0.0, 0.0, 0.10)
                    self.get_logger().info("Flying left")
                else:
                    self.fly(0.0, 0.0, 0.0, -0.10)
                    self.get_logger().info("Flying right")
            
            
            else:  # If it's near the center, move forward
                
                self.fly(0.2, 0.0, 0.0, 0.0)
                self.get_logger().info("Flying forward")
                time.sleep(0.1)  # waits for 10 seconds
                print("Moving on.")

            if width > 200 or height > 200:  # Adjust threshold as needed
                self.get_logger().info("Stop sign is close — initiating landing.")
                self.fly(0.0, 0.0, -1.0, 0.0)
                time.sleep(7)  # waits for 10 seconds
                #self.initiate_landing()

        else:
            self.fly(0.0, 0.0, 0.0, -0.12)
            self.get_logger().info("No gate - turning around")



        # Display result
        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 20, (0, 255, 0), 2)
        cv2.imshow("YOLOv11 Detection", frame)
        cv2.waitKey(1)

    def fly(self, lin_x, lin_y, lin_z, ang_z):
            fly = Twist()
            fly.linear.x = lin_x
            fly.linear.y = lin_y
            fly.linear.z = lin_z
            fly.angular.z = ang_z
            self.cmd_vel_pub.publish(fly)



def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
