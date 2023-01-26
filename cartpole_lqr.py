import mujoco as mj 
import numpy as np 
import argparse
from control import lqr
from tqdm import tqdm

xml_path = 'cartpole.xml'
sim_time = 10.

class Controller:
    def __init__(self,model,data):
        # Initialize the controller here.
        self.model = model
        self.data  = data
        
        self.n = 4
        self.m = 1
        
        self.A = np.zeros((self.n,self.n))
        self.B = np.zeros((self.n,self.m))
        
        self.linearize()
        
        self.Q = np.eye((self.n))
        self.R = 1e-2 * np.eye((self.m))
        
        self. K, _, _ = lqr(self.A, self.B, self.Q, self.R)
    
    def controller(self,model,data):
        x = np.array([data.qpos[0],data.qpos[1],data.qvel[0],data.qvel[1]])
        u = -self.K.dot(x)
        u = np.clip(u, -3., 3.)
        data.ctrl[0] = u

    def linearize(self):
        x0 = np.zeros((self.n,))
        u0 = np.zeros((self.m,))
        xdot0 = self.f(x0, u0)
        
        perturb = 1e-2
        
        for i in range(self.n):
            x = np.zeros_like(x0)
            u = u0
            x[i] += perturb
            xdot = self.f(x,u)
            for k in range(self.n):
                self.A[k,i] = (xdot[k] - xdot0[k])/perturb
        
        for i in range(self.m):
            x = x0
            u = np.zeros_like(u0)
            u[i] += perturb
            xdot=  self.f(x,u)
            for k in range(self.n):
                self.B[k,i] = (xdot[k] - xdot0[k])/perturb
    
    def f(self,x,u):
        self.data.qpos[0] = x[0]
        self.data.qpos[1] = x[1]
        self.data.qvel[0] = x[2]
        self.data.qvel[1] = x[3]
        self.data.ctrl[0] = u[0]
        mj.mj_forward(self.model, self.data)
        
        M = np.zeros((2,2))
        mj.mj_fullM(self.model,M,self.data.qM)
        Minv = np.linalg.inv(M)
        force_bias = np.array([self.data.qfrc_bias[0],self.data.qfrc_bias[1]])
        tau = np.array([u[0],0])
        qddot = np.matmul(Minv,np.subtract(tau,force_bias))
        
        xdot = np.array([self.data.qvel[0], self.data.qvel[1], qddot[0], qddot[1]])
        return xdot

if __name__ == '__main__':
    # Parse a seed from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--time', type=float, default=10.)
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    c = Controller(model,data)
    mj.set_mjcb_control(c.controller)

    sim_time = args.time

    for i in tqdm(range(args.num)):
        data.qpos[0] = np.random.uniform(-.5,.5)
        data.qpos[1] = np.random.uniform(-np.pi/6,np.pi/6)
        data.qvel[0] = 0.
        data.qvel[1] = 0.
        mj.mj_step(model,data)

        trajectory = [[data.qpos[0],data.qpos[1],data.qvel[0],data.qvel[1]]]

        while True:
            time_prev = data.time

            while (data.time - time_prev) < 1./60.:
                mj.mj_step(model,data)
                trajectory.append([data.qpos[0],data.qpos[1],data.qvel[0],data.qvel[1]])
            
            if data.time >= sim_time:
                sim_time += args.time
                break
        
        trajectory = np.array(trajectory)
        np.savetxt('data/trajectory_'+str(i)+'.txt',trajectory,delimiter=',')

