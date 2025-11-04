#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wheel_odom_node.py  [ROS1 / rospy]

/commands/motor/speed(ERPM)와 /commands/servo/position(0~1)을 직접 구독하여
Ackermann(자전거) 모델로 오도메트리를 적분해 /wheel_odom(nav_msgs/Odometry) 퍼블리시.

핵심 파라미터(rosparam YAML 등으로 주입):
  wheelbase [m], wheel_radius [m]
  odom_frame, base_frame, odom_topic
  publish_tf
  body_forward_sign
  use_imu_heading, imu_topic
  erpm_per_radps
  servo_center, max_steer_rad

메모:
  시작 직후에도 안전하게 돌도록 delta(조향각) 0.0으로 초기화
  IMU yaw가 오면 그 값을 우선 채택(드리프트 억제), 없으면 운동학 적분
  공분산 기본값 포함(필요시 조정)
"""

import math
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion


def q_to_yaw(x, y, z, w):
    """쿼터니언 전체를 사용해 yaw를 추출"""
    _, _, yaw = euler_from_quaternion([x, y, z, w])
    return yaw


class WheelOdomNode(object):
    def __init__(self):
        # 노드 파라미터
        self.L = rospy.get_param('~wheelbase', 0.325)
        self.R = rospy.get_param('~wheel_radius', 0.05)
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.odom_topic = rospy.get_param('~odom_topic', '/wheel_odom')
        self.publish_tf = rospy.get_param('~publish_tf', True)
        self.body_forward_sign = rospy.get_param('~body_forward_sign', 1.0)

        self.use_imu_heading = rospy.get_param('~use_imu_heading', True)
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_centered')

        self.rpm_scale = rospy.get_param('~rpm_scale', 0.025)  
        self.servo_center = rospy.get_param('~servo_center', 0.5)
        self.max_steer_rad = rospy.get_param('~max_steer_rad', 0.5)

        # 상태 변수
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_imu = None
        self.wz_imu = 0.0

        self.motor_speed = 0.0   # [rad/s]
        self.servo_pos = self.servo_center
        self.delta = 0.0

        # 퍼블리셔
        self.pub_odom = rospy.Publisher(self.odom_topic, Odometry, queue_size=20)

        # TF 브로드캐스터(옵션)
        self.tf_broadcaster = tf.TransformBroadcaster() if self.publish_tf else None

        # 서브스크라이버
        rospy.Subscriber('/commands/motor/speed', Float64, self.cb_motor, queue_size=20)
        rospy.Subscriber('/commands/servo/position', Float64, self.cb_servo, queue_size=20)
        if self.use_imu_heading:
            rospy.Subscriber(self.imu_topic, Imu, self.cb_imu, queue_size=50)

        # 50 Hz 타이머
        self.timer = rospy.Timer(rospy.Duration(1.0/100.0), self.on_timer)

        rospy.loginfo('wheel_odom node started [ROS1] (motor/servo inputs)')

    # 콜백들
    def cb_motor(self, msg):
        rpm = float(msg.data)*self.rpm_scale
        self.motor_speed = (rpm * 2.0 * math.pi) / 60.0  # rad/s

    def cb_servo(self, msg):
        self.servo_pos = float(msg.data)
        self.delta = (self.servo_pos - self.servo_center) * (2.0 * self.max_steer_rad)

    def cb_imu(self, msg):
        self.yaw_imu = q_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )
        self.wz_imu = float(msg.angular_velocity.z)

    # 타이머
    def on_timer(self, event):
        # dt 계산
        if event.last_real is None:
            dt = 0.0
        else:
            dt = (event.current_real - event.last_real).to_sec()
        if dt <= 0.0 or dt > 1.0:
            dt = 0.0

        self.update_odom(dt)

    # 핵심 로직
    def update_odom(self, dt):
        now_ros_time = rospy.Time.now()

        # 선속도 v = omega_wheel[rad/s] * R
        v_body = self.body_forward_sign * self.motor_speed * self.R

        # 요레이트 omega = v/L * tan(delta)
        omega_kin = (v_body * math.tan(self.delta) / self.L) if abs(self.L) > 1e-6 else 0.0

        # yaw 업데이트
        if dt > 0.0:
            if self.use_imu_heading and (self.yaw_imu is not None):
                self.yaw = self.yaw_imu
            else:
                self.yaw += omega_kin * dt

            # wrap to [-pi, pi]
            self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

            # 바디 전진을 월드(odom)로 투영
            self.x += v_body * math.cos(self.yaw) * dt
            self.y += v_body * math.sin(self.yaw) * dt

        # Odometry 메시지 구성
        odom = Odometry()
        odom.header.stamp = now_ros_time
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(self.yaw * 0.5)
        odom.pose.pose.orientation.w = math.cos(self.yaw * 0.5)

        odom.twist.twist.linear.x = v_body
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self.wz_imu if (self.use_imu_heading and self.yaw_imu is not None) else omega_kin

        # 공분산(예시)
        odom.pose.covariance = [
            0.05, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.05, 0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  1e6, 0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  1e6, 0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  1e6, 0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.1,
        ]
        odom.twist.covariance = [
            0.05, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.05, 0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  1e6, 0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  1e6, 0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  1e6, 0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.1,
        ]

        # 퍼블리시
        self.pub_odom.publish(odom)

        # 선택: TF 브로드캐스트
        if self.tf_broadcaster is not None:
            self.tf_broadcaster.sendTransform(
                (self.x, self.y, 0.0),
                (0.0, 0.0, math.sin(self.yaw * 0.5), math.cos(self.yaw * 0.5)),
                now_ros_time,
                self.base_frame,
                self.odom_frame
            )


def main():
    rospy.init_node('wheel_odom', anonymous=False)
    node = WheelOdomNode()
    rospy.spin()


if __name__ == '__main__':
    main()
