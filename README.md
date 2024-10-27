# mars-rover-vision
Computer vision for mars rover with MQTT broadcasting. Gif below shows our first full run (without obstacles) where the robot reaches all 6 red markers and the final end position. The glimmer of the lights above massively affected the ability to detect the ArUco code for live PID control of the robot, however even with brief glimpses it could correct itself in time.

![](https://github.com/Alexander-Evans-Moncloa/mars-rover-vision/blob/main/evidence.gif)

The image below shows the software detecting blue obstacles and rerouting the order of red obstacles, along with detecting a green ball and estimating the incoming angle of attack. These are all broadcasted via MQTT for the Raspberry Pi on board to detect and act accordingly.

![](https://github.com/Alexander-Evans-Moncloa/mars-rover-vision/blob/main/evidence.png)

