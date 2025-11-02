#!/usr/bin/env python3
import rospy
import numpy as np
import csv
from math import cos, sin, sqrt
from geometry_msgs.msg import Pose2D


class Pure_pursuit:
    def __init__(self, init_node: bool = False):
        if init_node:
            rospy.init_node("pure_pursuit_node", anonymous=True)
            rospy.Subscriber("/base_link_pose", Pose2D, self.base_callback)

        self.target_speed = 2.5 # 기본 목표 속도
        self.max_speed = 2.5
        self.min_speed = 0.5

        self.csv_file = "/root/KORA_K3/src/kora_k3/src/path_planning/outputs/waypoints.csv"
        self.waypoints = self.load_waypoints()
        self.curvature = 0.0

        # Lookahead 파라미터
        self.L0 = rospy.get_param("~L0", 0.6)
        self.k_v = rospy.get_param("~k_v", 0.5)
        self.Lmin = rospy.get_param("~Lmin", 0.5)
        self.Lmax = rospy.get_param("~Lmax", 1.0)

        self.v_est = 0.0
        self.alpha_v = rospy.get_param("~alpha_v", 0.3)
        self.lookahead_distance = max(self.Lmin, min(self.L0, self.Lmax))

    def base_callback(self, pose_msg):
        return self.compute_errors(pose_msg)

    def compute_errors(self, pose_msg, current_speed=None):
        """횡방향 오차(rad)와 목표 속도(m/s)를 계산."""
        if pose_msg is None:
            return None, None
        
        self.target_speed = (self.max_speed-self.min_speed)*np.exp(-(abs(self.curvature)))+self.min_speed
        print(self.target_speed)

        if current_speed is None:
            current_speed = self.target_speed

        self.v_est = (1.0 - self.alpha_v) * self.v_est + self.alpha_v * current_speed
        Ld = self.L0 + self.k_v * abs(self.v_est)
        self.lookahead_distance = max(self.Lmin, min(Ld, self.Lmax))

        goal_point = self.find_goal_point(pose_msg)
        if not goal_point:
            goal_point = self.fallback_forward_point(pose_msg)
        if not goal_point:
            return None, None

        steering_error = self.calculate_steering_angle(goal_point)
        steering_error = float(np.clip(steering_error, -0.5, 0.5))

        return steering_error, self.target_speed

    def find_goal_point(self, odom_msg):
        # 현재 차량 위치
        car_x = odom_msg.x                    # car x
        car_y = odom_msg.y                       # car y
        yaw = odom_msg.theta                  # car yaw

        # 목표 경로점 리스트 초기화
        max_distance = -1
        goal_point = []
        
        for x, y in self.waypoints:
            # 차량과 경로점 간의 거리 계산
            dx = x - car_x
            dy = y - car_y
            distance = sqrt(dx**2 + dy**2)

            # 거리 조건을 먼저 확인 (차량 앞쪽)
            if distance <= self.lookahead_distance:
                # 차량 프레임에서 x축이 양수인 경우만 앞쪽으로 간주
                rotated_x = cos(-yaw) * dx - sin(-yaw) * dy
                rotated_y = sin(-yaw) * dx + cos(-yaw) * dy

                if rotated_x > 0:
                    # 가장 먼 점을 실시간으로 찾기
                    if distance > max_distance:
                        max_distance = distance
                        goal_point = (x, y, rotated_x, rotated_y, distance)

        return goal_point
    
    def fallback_forward_point(self, pose_msg):
        # lookahead 안에서 전방점을 못 찾았을 때 대비(가장 가까운 '앞' 점)
        car_x = pose_msg.x                    # car x
        car_y = pose_msg.y                    # car y
        yaw = pose_msg.theta  

        best = None
        best_d = float('inf')
        for x, y in self.waypoints:
            dx, dy = x - car_x, y - car_y
            # 차량 프레임 x>0만 전방
            fx = cos(-yaw) * dx - sin(-yaw) * dy
            if fx <= 0:
                continue
            d = sqrt(dx*dx + dy*dy)
            if d < best_d:
                # 차량 프레임 y도 함께 저장
                fy = sin(-yaw) * dx + cos(-yaw) * dy
                best = (x, y, fx, fy, d)
                best_d = d
        return best 

    def calculate_steering_angle(self, goal_point):
        # Pure Pursuit: curvature = 2*y / L^2  (여기서 y는 차량 프레임에서의 lateral)
        L = max(1e-3, self.lookahead_distance)  # 0 방지
        y_err = goal_point[3]
        self.curvature = 2.0 * y_err / (L * L)
        return self.curvature


    def load_waypoints(self):
        """CSV 파일에서 웨이포인트를 읽어 리스트로 반환"""
        waypoints = []
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 건너뛰기 (필요 시)
            for row in reader:
                # x, y 좌표 추출
                x = float(row[0])
                y = float(row[1])
                waypoints.append((x, y))
        return waypoints

# def main():
#     try:
#         pure_pursuit = Pure_pursuit()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass

# if __name__ == "__main__":
#     main()
