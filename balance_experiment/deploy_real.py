import time
import sys
from dotenv import load_dotenv
import os
import select
import tty
import termios
import numpy as np
load_dotenv()
NETWORK_CARD_NAME = os.getenv('NETWORK_CARD_NAME')
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

from utils_real import init_cmd_hg, create_damping_cmd, create_zero_cmd, MotorMode, NonBlockingInput, joint2motor_idx,Kp,Kd, G1_NUM_MOTOR, default_pos


class Controller:
    def __init__(self) -> None:


        # Initialize the policy network
        # self.policy = torch.jit.load(config.policy_path)
        # # Initializing process variables
        # self.qj = np.zeros(config.num_actions, dtype=np.float32)
        # self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        # self.action = np.zeros(config.num_actions, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

      


        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def send_cmd(self, cmd:  LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)
   
    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Press Enter to continue...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'): 
                create_zero_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd) 
        print("Zero torque state confirmed. Proceeding...")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
 
        # default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        
        
        # record the current pos
        init_dof_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press Enter to start the controller...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'): # Check for Enter key
                # Keep sending default position commands while waiting
                for i in range(len(joint2motor_idx)):
                    motor_idx = joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = default_pos[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
               
                self.send_cmd(self.low_cmd) 
                time.sleep(self.config.control_dt) 
        print("Default position state confirmed. Starting controller...")


if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # Initial prompt doesn't need non-blocking
    input("Press Enter to acknowledge warning and proceed...")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller()

    # Enter the zero torque state, press Enter key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press Enter key to continue executing
    controller.default_pos_state()

    print("Controller running. Press 'q' to quit.")
    with NonBlockingInput() as nbi: # Use context manager for the main loop
        while True:
            controller.run()
            # Check for 'q' key press to exit
            if nbi.check_key('q'):
                print("\n'q' pressed. Exiting loop...")
                break
            # Add a small sleep to prevent busy-waiting if controller.run() is very fast
            time.sleep(0.001)


    print("Entering damping state...")
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
        
    print("Exit")
