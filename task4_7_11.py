#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8, HumanPoseEstimation
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from gtts import gTTS
from playsound import playsound
from RobotChassis import RobotChassis
import datetime
from std_srvs.srv import Empty

from typing import Tuple, List


class FollowMe(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])

        if len(p) == 0:
            return -1, -1, -1
        return int(p[0][0]), int(p[0][1]), 1

    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0:
            return 0, 0, 0
        a1 = 55.0
        b1 = 86.0
        a = a1 * np.pi / 180
        b = b1 * np.pi / 180

        d = depth[y][x]
        h, w = depth.shape[:2]
        if d == 0:
            for k in range(1, 15, 1):
                if d == 0 and y - k >= 0:
                    for j in range(x - k, x + k, 1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y - k][j]
                        if d > 0:
                            break
                if d == 0 and x + k < w:
                    for i in range(y - k, y + k, 1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x + k]
                        if d > 0:
                            break
                if d == 0 and y + k < h:
                    for j in range(x + k, x - k, -1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y + k][j]
                        if d > 0:
                            break
                if d == 0 and x - k >= 0:
                    for i in range(y + k, y - k, -1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x - k]
                        if d > 0:
                            break
                if d > 0:
                    break
        x = x - w // 2
        y = y - h // 2
        real_y = y * 2 * d * np.tan(a / 2) / h
        real_x = x * 2 * d * np.tan(b / 2) / w
        return real_x, real_y, d

    def calc_linear_x(self, cd: float, td: float) -> float:
        if cd <= 0:
            return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0:
            x = min(x, 0.15)
        if x < 0:
            x = max(x, -0.15)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0:
            return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0:
            z = min(z, 0.2)
        if z < 0:
            z = max(z, -0.2)
        return z

    def calc_cmd_vel(self, image, depth, cx, cy) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()

        frame = image
        if cx == 2000:
            cur_x, cur_z = 0, 0
            return cur_x, cur_z, frame, "no"

        print(cx, cy)
        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 800)
        cur_z = self.calc_angular_z(cx, 320)

        dx = cur_x - self.pre_x
        if dx > 0:
            dx = min(dx, 0.15)
        if dx < 0:
            dx = max(dx, -0.15)

        dz = cur_z - self.pre_z
        if dz > 0:
            dz = min(dz, 0.4)
        if dz < 0:
            dz = max(dz, -0.4)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x
        self.pre_z = cur_z

        return cur_x, cur_z, frame, "yes"


# gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_imu(msg):
    global _imu
    _imu = msg


def get_distance(px, py, pz, ax, ay, az, bx, by, bz):
    A, B, C, p1, p2, p3, qx, qy, qz, distance = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    A = int(bx) - int(ax)
    B = int(by) - int(ay)
    C = int(bz) - int(az)
    p1 = int(A) * int(px) + int(B) * int(py) + int(C) * int(pz)
    p2 = int(A) * int(ax) + int(B) * int(ay) + int(C) * int(az)
    p3 = int(A) * int(A) + int(B) * int(B) + int(C) * int(C)
    # print("1",p1,p2,p3)
    if (p1 - p2) != 0 and p3 != 0:
        t = (int(p1) - int(p2)) / int(p3)
        qx = int(A) * int(t) + int(ax)
        qy = int(B) * int(t) + int(ay)
        qz = int(C) * int(t) + int(az)
        return int(int(pow(((int(qx) - int(px)) ** 2 + (int(qy) - int(py)) ** 2 + (int(qz) - int(pz)) ** 2), 0.5)))
    return 0


def get_real_xyz(dp, x, y, s):
    numy,numz=0,0
    if s=="down": numy,numz=-250,187
    else: numy,numz=-1050,80
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    print(real_x, real_y)
    return real_x, real_y+numy, d+numz


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0: return -1, -1
    return int(p[0][0]), int(p[0][1])


def say(g):
    tts = gTTS(g)

    # Save the speech as an audio file
    speech_file = "speech.mp3"
    tts.save(speech_file)

    # Play the speech
    playsound(speech_file)


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def set_gripper(angle, t):
    service_name = "/goal_tool_control"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        request.joint_position.position = [angle]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def set_joints(joint1, joint2, joint3, joint4, t):
    service_name = "/goal_joint_space_path"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = [
            "joint1", "joint2", "joint3", "joint4"]
        request.joint_position.position = [joint1, joint2, joint3, joint4]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def open_gripper(t):
    return set_gripper(0.01, t)


def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 0.2
    limit_time = 3
    start_time = rospy.get_time()
    while True:
        q = [
            _imu.orientation.x,
            _imu.orientation.z,
            _imu.orientation.y,
            _imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
        print(yaw, e)
        if yaw < 0 and angle > 0:
            cw = np.pi + yaw + np.pi - angle
            aw = -yaw + angle
            if cw < aw:
                e = -cw
        elif yaw > 0 and angle < 0:
            cw = yaw - angle
            aw = np.pi - yaw + np.pi + angle
            if aw < cw:
                e = aw
        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break
        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()
    move(0.0, 0.0)


def turn(angle: float):
    global _imu
    q = [
        _imu.orientation.x,
        _imu.orientation.y,
        _imu.orientation.z,
        _imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.1)


def close_gripper(t):
    return set_gripper(-0.01, t)


def calc_linear_x(cd, td):
    if cd <= 0: return 0
    e = cd - td
    p = 0.0005
    x = p * e
    if x > 0: x = min(x, 0.5)
    if x < 0: x = max(x, -0.5)
    return x


def calc_angular_z(cx, tx):
    if cx < 0: return 0
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z


def move_to(x, y, z, t):
    service_name = "/goal_task_space_path_position_only"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetKinematicsPose)

        request = SetKinematicsPoseRequest()
        request.end_effector_name = "gripper"
        request.kinematics_pose.pose.position.x = x
        request.kinematics_pose.pose.position.y = y
        request.kinematics_pose.pose.position.z = z
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def callback_voice(msg):
    global s
    s = msg.text


def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def test_point(xs, ys, d):
    sty = ys
    stx = xs
    d = d * np.pi / 180
    if sty * math.sin(d) < 0:
        n1x = -abs(sty * math.sin(d)) + stx * math.cos(d)
    else:
        n1x = abs(sty * math.sin(d)) + stx * math.cos(d)

    n1y = ((stx ** 2 + sty ** 2) * math.cos(d) - stx * n1x) / sty
    return n1x, n1y


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    frame2 = None
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth2)

    _image1 = None
    _topic_image1 = "/cam2/color/image_raw"
    frame1 = rospy.Subscriber(_topic_image1, Image, callback_image1)
    print("astra depth")
    _depth1 = None
    _topic_depth1 = "/cam2/depth/image_raw"
    depth1 = rospy.Subscriber(_topic_depth1, Image, callback_depth1)
    # _sub_up_cam_image.unregister()

    s = ""
    t = 3.0
    print("speaker")
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    dnn_yolo1 = Yolov8("bag_100", device_name="GPU")
    dnn_yolo1.classes = ['obj']
    print("yolo")

    action = "none"
    net_pose = HumanPoseEstimation(device_name="GPU")
    step = "person"  # remember
    f_cnt = 0
    step2 = "dead"  # remember
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    b1, b2, b3, b4 = 0, 0, 0, 0
    pre_z, pre_x = 0, 0
    cur_z, cur_x = 0, 0
    test = 0
    get1 = 0
    p_list = []
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu = None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)
    say("start the program")
    mode = 0
    sb = 0
    chassis = RobotChassis()
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    say_degree = 0
    framecnt = 0
    bottlecnt = 0
    # detector1 = ColorDetector((170, 16, 16), (190, 255, 255)
    bottlecolor = ["pink", "black", "yellow"]
    saidd = 0
    get_b = 0
    dis_list = []
    # depth up 70 down 240
    open_gripper(3)

    line_destory_cnt = 0
    action = "none"
    b1,b2,b3=0,0,0
    hand_cnt = 0
    bottle = []
    mark=0
    s_c, s_d = [], []
    check_bootle_cnt = 0
    _fw = FollowMe()
    hl=0

    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if frame2 is None:
            print("frame_down")
            continue
        if depth2 is None:
            print("depth_down")
            continue
        if frame1 is None:
            print("frame_up")
            continue
        if depth1 is None:
            print("depth_up")
            continue

        down_image = frame2.copy()
        down_depth = depth2.copy()
        up_image = _image1.copy()
        up_depth = _depth1.copy()
        if step == "ft":
            clear_costmaps
            say("start")
            chassis.move_to(-0.857, -0.381, -2.719)
            # 門口
            # checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            say("Master, your son got choked and he seems to be very painful")
            clear_costmaps
            chassis.move_to(3.93, -0.7740, 2.045)
            # 原點
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            time.sleep(1)
            print("next")
            step = "person"
        if step == "person" and ("help" in s or "carry" in s):
            az, bz = 0, 0
            A = []
            B = []
            yu = 0
            poses = net_pose.forward(up_image)
            if len(poses) > 0:
                YN = -1
                a_num, b_num = 9, 7
                for issack in range(len(poses)):
                    yu = 0
                    if poses[issack][9][2] > 0 and poses[issack][7][2] > 0:
                        YN = 0

                        a_num, b_num = 9, 7
                        A = list(map(int, poses[issack][a_num][:2]))
                        if (640 >= A[0] >= 0 and 320 >= A[1] >= 0):
                            ax, ay, az = get_real_xyz(up_depth, A[0], A[1],"up")
                            if az <= 1800 and az != 0:
                                yu += 1
                        B = list(map(int, poses[issack][b_num][:2]))
                        if (640 >= B[0] >= 0 and 320 >= B[1] >= 0):
                            bx, by, bz = get_real_xyz(up_depth, B[0], B[1],"up")
                            if bz <= 1800 and bz != 0:
                                yu += 1
                    if yu >= 2:
                        break
            print(A, B)
            if len(A) != 0 and len(B) != 0 and yu >= 2:
                cv2.circle(up_image, (A[0], A[1]), 3, (255, 255, 0), -1)
                cv2.circle(up_image, (B[0], B[1]), 3, (255, 255, 0), -1)
            if len(A) != 0 and len(B) != 0 and az != 0 and bz != 0:
                hand_cnt += 1
            if hand_cnt >= 10:
                # time.sleep(2)
                print("before position", ax, ay, az, bx, by, bz)
                step = "_code"
                # turn(degree666)
                say("I see your hand")
                move(0.0, 0.0)
                az+=80
                bz+=80
                print("before position", ax, ay, az, bx, by, bz)
        if step == "_code":
            if mode == 0 and mode != 1:
                q = [
                    _imu.orientation.x,
                    _imu.orientation.y,
                    _imu.orientation.z,
                    _imu.orientation.w
                ]
                roll1, pitch1, yaw1 = euler_from_quaternion(q)
                mode = 1
            # break
            al = []
            detections = dnn_yolo1.forward(down_image)[0]["det"]
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                score = detection[4]
                if class_id != 0: continue
                if score < 0.5: continue
                if cy >= 480: continue
                print(cx, cy, "position bottle")
                cx = min(cx, 640)
                cy = min(cy, 480)
                if (640 <= cx or cx < 0 and 320 <= cy or cy < 0): continue

                k2, kk1, kkkz = get_real_xyz(down_depth, cx, cy,"down")
                if kkkz > 2500 or abs(kkkz) <= 0: continue
                al.append([x1, y1, x2, y2, score, class_id])
                # print(float(score), class_id)
                hhh = str(class_id) + " " + str(k2) + " " + str(kk1) + " " + str(kkkz)
                cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.putText(down_image, str(hhh), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bb = sorted(al, key=(lambda x: x[0]))

            if (len(bb) < 3) and mode == 1:
                move(0, 0.3)
            else:
                if hl==0:
                    step2="yyyy"
                hl+=1
            
            if step2 == "yyyy":
                mode += 1
                if get1 == 0:
                    mode = 3
                    q = [
                        _imu.orientation.x,
                        _imu.orientation.y,
                        _imu.orientation.z,
                        _imu.orientation.w
                    ]
                    roll2, pitch2, yawb = euler_from_quaternion(q)
                    print("hello")
                    get1 += 1

                yaw2 = yawb
                move(0, 0)
                degree666 = (2 * np.pi - ((yaw2 + np.pi) - (yaw1 + np.pi))) * 180 / np.pi % 360
                degree666 = 360 - degree666
                print("turn_degreeeeeeeeeeeee:  ", degree666)
                if say_degree == 0:
                    say("I turn" + str(int(degree666)) + "degree")
                    say_degree += 1
                print("yaw1-yaw2", yaw1 - yaw2)
                print("yaw1", yaw1)
                print("yaw2", yaw2)
                print("roll1", roll1)
                print("roll2", roll2)
                print("pitch1", pitch1)
                print("pitch2", pitch2)
                axs2, azs2 = test_point(ax, az , degree666)
                bxs2, bzs2 = test_point(bx, bz, degree666)
                print("before position", ax, ay, az, bx, by, bz)
                print("cal hand position", axs2, ay, azs2, bxs2, by, bzs2)
                for i, detection in enumerate(bb):
                    # print(detection)
                    x1, y1, x2, y2, score, class_id = map(int, detection)
                    score = detection[4]
                    # print(id)
                    ggg = 1
                    bottle.append(detection)
                    E += 1
                    cx1 = (x2 - x1) // 2 + x1
                    cy1 = (y2 - y1) // 2 + y1

                    px, py, pz = get_real_xyz(down_depth, cx1, cy1,"down")
                    dis_list.append(pz)
                    # pxs2, pzs2, pxs1, pzs1 = test_point(px, pz, degree666)
                    # print("bottle",i+1, pxs2,py, pzs2)
                    cnt = get_distance(px, py, pz, axs2, ay, azs2, bxs2, by, bzs2)
                    cv2.putText(down_image, str(int(cnt) // 10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                                (0, 0, 255), 2)
                    cnt = int(cnt)
                    '''
                    if cnt != 0 and cnt <= 600:
                        cnt = int(cnt)
                    else:
                        cnt = 9999'''
                    s_c.append(cnt)
                    s_d.append(pz)

                if ggg == 0: s_c = [9999]
                TTT = min(s_c)
                E = s_c.index(TTT)
                for i, detection in enumerate(bottle):
                    # print("1")
                    x1, y1, x2, y2, score, class_id = map(int, detection)

                    if (class_id == 0):
                        if i == E:
                            cx1 = (x2 - x1) // 2 + x1
                            cy1 = (y2 - y1) // 2 + y1

                            cv2.putText(down_image, str(int(TTT) // 10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.15,
                                        (0, 0, 255), 2)
                            cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            _, _, dddd1 = get_real_xyz(down_depth, cx1, cy1,"down")
                            if i == 0: b1 += 1
                            if i == 1: b2 += 1
                            if i == 2: b3 += 1
                            
                            print("i",i)
                            print("b1",b1,"b2",b2,"b3",b3)
                        else:
                            v = s_c[i]
                            cv2.putText(down_image, str(int(v)), (x1 + 5, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)
            print("b1",b1,"b2",b2,"b3",b3)
            if b1 == max(b1, b2, b3): mark = 0
            if b2 == max(b1, b2, b3): mark = 1
            if b3 == max(b1, b2, b3): mark = 2
            if (b1 >= 1 or b2 >= 1 or b3 >= 1) and sb==0:
                step2 = "gettttt"
            if step2 == "gettttt":
                # mark = 0
                if len(bb)<(mark+1):
                    mark=max(len(bb)-1,0)
                print("turn_get")
                if sb == 0:
                    if mark == 0: say("the left bag")
                    if mark == 1: say("the middle bag")
                    if mark == 2: say("the right bag")
                    sb += 1
                print(bb)
                h, w, c = down_image.shape
                x1, y1, x2, y2, score, class_id = map(int, bb[mark])
                '''
                if framecnt == 0:
                    face_box = [x1, y1, x2, y2]
                    box_roi = down_depth[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                    fh, fw = abs(x1 - x2), abs(y1 - y2)
                    box_roi = cv2.resize(box_roi, (fh * 10, fw * 10), interpolation=cv2.INTER_AREA)
                    cv2.imshow("bottle", box_roi)
                    get_b = mark
                    framecnt += 1'''
                cx2 = (x2 - x1) // 2 + x1
                cy2 = (y2 - y1) // 2 + y1
                e = w // 2 - cx2
                v = 0.0015 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                move(0, v)
                if abs(e) <= 5:
                    step2 = "go"
            if step2 == "go":

                cx, cy = w // 2, h // 2
                for i in range(cy + 1, h):
                    if down_depth[cy][cx] == 0 or 0 < down_depth[i][cx] < down_depth[cy][cx]:
                        cy = i
                _, _, d = get_real_xyz(down_depth, cx, cy,"down")
                timems = round(abs(d*0.5)) // 0.2 // 10 -5
                print(timems)
                for i in range(round(timems)):
                    move(0.2, 0)
                    time.sleep(0.01)
                step2 = "none"
                step="grap"
                say("turn again")
            if step2 == "turn_again":
                print("mode eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeturn again")
                if len(bb) < 1: continue
                print("turnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
                min11 = 999999
                ucx, ucy = 0, 0
                for num in bb:
                    x1, y1, x2, y2, score, class_id = map(int, num)
                    cx2 = (x2 - x1) // 2 + x1
                    cy2 = (y2 - y1) // 2 + y1
                    _, _, d = get_real_xyz(down_depth, cx2, cy2,"down")
                    if abs(w // 2 - cx2) < min11:
                        min11 = abs(w // 2 - cx2)
                        ucx = cx2

                h, w, c = down_image.shape
                e = w // 2 - ucx
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                move(0, v)
                if abs(e) <= 3:
                    step = "grap"
                    step2 = "none"

        if step == "grap":
            print("got there")
            open_gripper(t)
            joint1, joint2, joint3, joint4 = 0.0, 1.104, 0.758, -1.7
            set_joints(joint1, joint2, joint3, joint4, 1)
            time.sleep(3)
            say("get it")
            e = 1
            if abs(e) <= 3:
                for i in range(39000): move(0.2, 0)

                say("I get it")
                time.sleep(t)

                close_gripper(t)
                time.sleep(2)
                joint1, joint2, joint3, joint4 = 0.00, 1.104, 0.758, -1.7
                set_joints(joint1, joint2, joint3, joint4, 1)
                time.sleep(t)

                joint1, joint2, joint3, joint4 = -0.106, 0.419, 0.365, -1.4
                set_joints(joint1, joint2, joint3, joint4, t)
                time.sleep(t)
                joint1, joint2, joint3, joint4 = 0, 0, 0, -1.0
                set_joints(joint1, joint2, joint3, joint4, t)
                time.sleep(t)
                time.sleep(3)
                say("I will follow you now")
                for i in range(130000): move(-0.2, 0)
            time.sleep(2)
            step = "givehim"

        if step == "givehim":
            clear_costmaps
            say("I got the bag")
            chassis.move_to(1.76, -0.196, 2.921)
            # 門口
            # checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            step = "follow"
        if step == "follow":
            print('follow')
            msg = Twist()
            poses = net_pose.forward(up_image)
            min_d = 9999
            t_idx = -1
            for i, pose in enumerate(poses):
                if pose[5][2] == 0 or pose[6][2] == 0:
                    continue
                p5 = list(map(int, pose[5][:2]))
                p6 = list(map(int, pose[6][:2]))

                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2
                cv2.circle(up_image, p5, 5, (0, 0, 255), -1)
                cv2.circle(up_image, p6, 5, (0, 0, 255), -1)
                cv2.circle(up_image, (cx, cy), 5, (0, 255, 0), -1)
                _, _, d = get_real_xyz(up_depth, cx, cy,"up")
                if d >= 1800 or d == 0: continue
                if (d != 0 and d < min_d):
                    t_idx = i
                    min_d = d

            x, z = 0, 0
            if t_idx != -1:
                p5 = list(map(int, poses[t_idx][5][:2]))
                p6 = list(map(int, poses[t_idx][6][:2]))
                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(up_depth, cx, cy,"up")
                cv2.circle(up_image, (cx, cy), 5, (0, 255, 255), -1)

                print("people_d", d)
                if d >= 1800 or d == 0: continue

                x, z, up_image, yn = _fw.calc_cmd_vel(up_image, up_depth, cx, cy)
                print("turn_x_z:", x, z)
            move(x, z)
            action = "check_voice"
        if step == "follow" and action == "check_voice":
            s = s.lower()
            print("speak", s)
            if "thank" in s or "you" in s or "ok" in s or "robot" in s:
                action = "none"
                say("No worries, take care")
                joint1, joint2, joint3, joint4 = 0.000, 0.0, 0, 1.5
                set_joints(joint1, joint2, joint3, joint4, 1)
                time.sleep(t)
                open_gripper(t)
                time.sleep(3)
                joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3, 0.70
                set_joints(joint1, joint2, joint3, joint4, 1)

                time.sleep(2.5)
                joint1, joint2, joint3, joint4 = 1.7, -1.052, 0.376, 0.696
                set_joints(joint1, joint2, joint3, joint4, 3)
                action = "none"
                step = "back3"
        if step == "back3":
            clear_costmaps
            chassis.move_to(3.467, -1.070, 2.045)
            # 原點
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            say("I am back")
            break
        E = down_image.copy()
        h, w, c = up_image.shape
        upout = cv2.line(up_image, (320, 0), (320, 500), (0, 255, 0), 5)
        downout = E.copy()
        img = np.zeros((h, w * 2, c), dtype=np.uint8)
        img[:h, :w, :c] = upout
        img[:h, w:, :c] = downout
        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
