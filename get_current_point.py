#!/usr/bin/env python3
import rospy
from RobotChassis import RobotChassis


if __name__ == "__main__":
    rospy.init_node("home_edu_robot_chassis")
    rate = rospy.Rate(20)
    
    chassis = RobotChassis()
    
    P = chassis.get_current_pose()
    rospy.loginfo("From %.2f, %.2f, %.2f" % (P[0], P[1], P[2]))
    
    while not rospy.is_shutdown():
        #clear_costmaps
        # chassis.move_to(-0.531,-2.99,0.201)
        chassis.move_to(-6.96, -6.2, 0)
        # checking
        while not rospy.is_shutdown():
            # 4. Get the chassis status.
            code = chassis.status_code
            text = chassis.status_text
            if code == 3:
                break
        P = chassis.get_current_pose()
        print("From %.2f, %.2f, %.2f" % (P[0], P[1], P[2]))        
        x1,x2,y1,y2=-5.68,-6.38,-6.13,-7.57
        P1,P2,P3,P4=[y1,x1],[y1,x2],[y2,x1],[y2,x2]
        
        #left up, right up, left down, right down [y,x]
        
        if((P[0]<=y1 and P[0]>=y2) and (P[1]<=x1 and P[1]>=x2)):
        
            print("I am there")
            break
            
        
        
