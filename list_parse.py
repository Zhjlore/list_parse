
import torch
import numpy as  np
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import sys,os

# torch.compile 需要在安装 pytorch2.0 之后方可使用

#  1 simulation time unit = 100ms - dt


class Cognitive:
    def __init__(self, dt=0.1, input_num=10, chunk_num=30, device ='cuda:0'):
        self.device = device
        self.dt = dt   # 0.1

        self.chunk_num=chunk_num

        self.Xi = torch.zeros(input_num, dtype=torch.float32).to(self.device)   # layer4 i通道细胞的活动
        self.Xi_old = self.Xi.clone()
        self.Xk = torch.zeros(input_num, dtype=torch.float32).to(self.device)   # layer4 k通道细胞的活动
        self.Xk_old = self.Xk.clone()

        self.Yi = torch.zeros(input_num, dtype=torch.float32).to(self.device)   # layer5/6 i通道细胞的活动
        self.Yi_old = self.Yi.clone()
        self.Yk = torch.zeros(input_num, dtype=torch.float32).to(self.device)   # layer5/6 k通道细胞的活动
        self.Yk_old = self.Yk.clone()

        self.Cj = torch.zeros(chunk_num, dtype=torch.float32).to(self.device)   # layer2/3 第j个列表块的活动
        self.Cj_old = self.Cj.clone()

        self.Wij = 1.1*torch.rand((input_num,chunk_num), dtype=torch.float32).to(self.device)  # bottom-up 自适应权重
        self.Wij_old = self.Wij.clone()
        self.Mji = 1.1*torch.rand((chunk_num,input_num), dtype=torch.float32).to(self.device)  # top-down weight
        self.Mji_old = self.Mji.clone()

    def run(self, I, K, J):
        Y = self.forward(I, K, J)
        self.update()
        return Y

    def update(self):
        self.Xi_old = self.Xi.clone()
        self.Xk_old = self.Xk.clone()
        self.Yi_old = self.Yi.clone()
        self.Yk_old = self.Yk.clone()
        self.Cj_old = self.Cj.clone()
        self.Wij_old = self.Wij.clone()
        self.Mji_old = self.Mji.clone()

    @torch.compile
    def forward(self, I, K, J):
        """
        :param K: 细胞Ck接收的输入数
        :param J: 细胞Cj接收的输入数
        """
        # layer4 细胞的活动  # F = 1.25 测量意志增益控制的效果  e = 0.05 layer4 人类认知数据模拟
        self.Xi = self.Xi_old + self.dt * (-0.1 * self.Xi_old + (1 - self.Xi_old) * (I + 0.05 * fun.relu(self.Yi_old))     # [10]
                                           - 1.25 * self.Xi_old * (torch.sum(I + 0.05 * fun.relu(self.Yi_old), dim=0)
                                           - (I + 0.05 * fun.relu(self.Yi_old))))

        # print("ss:",self.Xi,self.Xi_old,self.Yi_old,I)
        f1_Cj = torch.pow(self.Cj_old, 2) / (1 + torch.pow(self.Cj_old, 2))                                     # [30]
        # layer5/6 i通道细胞活动  b = 0.7  layer6 人类认知数据
        self.Yi = self.Yi_old + self.dt * (-0.1 * self.Yi_old + (1 - self.Yi_old) * (                                       # [10]
                                            I + 0.7 * fun.relu(self.Xi_old) + (f1_Cj @ self.Mji_old)))


        # f1_Ck = f1_Cj
        # J K --> 细胞Cj和Ck接收的输入数  K^J ---> 细胞Cj和Ck共享的输入数
        g_Cj=torch.pow(self.Cj_old, 2) / (16 + torch.pow(self.Cj_old, 2))
        def func(K, J):  # (30, 30)
            return (torch.sum((g_Cj * K * (1+J)), dim=0)-(g_Cj * K * (1+J)))/((self.chunk_num-1)*(K * (1 + J)))

        # layer2/3 第j个列表块的活动
        #expanded_Xi_old = self.Xi_old.unsqueeze(0)   # [1, 10]
        self.Cj = self.Cj_old + self.dt * (-0.1 * self.Cj_old + (1 - self.Cj_old) * (            # [30]
                20 / (10 + J) * (fun.relu(self.Xi_old) @ self.Wij_old)
                + 0.001 * J * f1_Cj) - 2 * (1 + self.Cj_old) * func(K, J))
        # print((torch.sum((f1_Cj * K * (1+J)), dim=0)-(f1_Cj * K * (1+J))))



        f2_Cj=f1_Cj
        Xi_old = self.Xi_old.view(10, 1)
        # bottom-up 自适应权重Wij - 初始值(0.001) # a1 = 1  h1 = 2  d1 = 1  h1 = 2
        self.Wij = self.Wij_old + self.dt * (f2_Cj * ((1 - self.Wij_old) * Xi_old - 2 * self.Wij_old * (    # [10, 30]
                    torch.sum(Xi_old, dim=0) - Xi_old)))
        # print(self.Wij)  # [10, 30]

        Yi_repeat = self.Yi_old.unsqueeze(0).repeat(self.chunk_num, 1)  # [30, 10]
        f2_Cj_repeat = f2_Cj.unsqueeze(1)                   # [30, 1]
        # top-down weight Mji 初始值为0.001  # a2 = 1

        self.Mji = self.Mji_old + self.dt * f2_Cj_repeat * (Yi_repeat - self.Mji_old)      # [30, 10]

        return self.Yi

class Motor:
    def __init__(self, num, dt, device='cuda:0'):
        self.dt = dt
        self.device = device

        self.Fi = torch.zeros(num, dtype=torch.float32).to(self.device)   # motor 行动计划区域细胞
        self.Fi_old = self.Fi.clone()

        self.Si = torch.zeros(num, dtype=torch.float32).to(self.device)   # plan selection 计划选择区域细胞
        self.Si_old = self.Si.clone()

        self.Qi = torch.zeros(num, dtype=torch.float32).to(self.device)   # 抑制性中间神经元活动
        self.Qi_old = self.Qi.clone()

    def run(self, G, R, Yi, Ei):
            S = self.forward(G, R, Yi, Ei)
            self.update()
            return S

    def update(self):
            self.Fi_old = self.Fi.clone()
            self.Si_old = self.Si.clone()
            self.Qi_old = self.Qi.clone()

    @torch.compile
    def forward(self, G, R, Yi, Ei):
        """
        :param G: GO信号
        :param R: 调节信号
        :param Yi: cognitive输入信息
        :param Ei: 注意增强信号
        """

        f3_Fi = torch.pow(self.Fi_old, 1.2)/(0.8**1.2+torch.pow(self.Fi_old, 0.2))  # [10]
        f4_G = torch.pow(G, 2)/(0.02*0.02+torch.pow(G, 2))  # [1]
        f5_Yi = fun.relu(Yi - 0.165)  # [10]
        f3_Qk = torch.pow(self.Qi_old, 1.2)/(0.8**1.2+torch.pow(self.Qi_old, 0.2))  # [10]
        # print(f3_Fi.shape,f4_G.shape,f5_Yi.shape,f3_Qk.shape)

        # motor 行动计划区域细胞  # [10]
        self.Fi = self.Fi_old + self.dt*(-0.1 * self.Fi_old + (1-self.Fi_old)*((1+G)*f3_Fi + Ei + (1-f4_G) * f5_Yi)       # [10]
                                        - self.Fi_old * (100*fun.relu(self.Si_old-0.5) + (1+G)*(torch.sum(f3_Qk, dim=0)-f3_Qk)
                                        + torch.sum(Ei, dim=0) - Ei + (1-f4_G) * (torch.sum(f5_Yi)-f5_Yi)))
        # print(self.Fi.shape)
        # 抑制性中间神经元活动
        self.Qi = self.Qi_old + self.dt * (self.Fi_old - self.Qi_old)  # [10]

        # 计划选择区域细胞  # [10]
        self.Si = self.Si_old + self.dt * (-0.1*self.Si_old + (1-self.Si_old)*(fun.relu(R) * self.Fi_old + fun.relu(self.Si_old - 0.5))  # [10]
                                           - 20*self.Si_old * (torch.sum(fun.relu(self.Si_old-0.5), dim=0)-fun.relu(self.Si_old-0.5)))

        return self.Si

class VITE:
    def __init__(self, num, dt, device='cuda:0'):
        self.device = device
        self.dt = dt

        # 意志门控信号G
        self.G = torch.zeros(1, dtype=torch.float32).to(self.device)
        self.G_old = self.G.clone()
        # 目标位置向量（TPV）
        self.T = torch.zeros(num, dtype=torch.float32).to(self.device)
        self.T_old = self.T.clone()
        # 位置差分向量（DV)
        self.D = torch.zeros(num, dtype=torch.float32).to(self.device)
        self.D_old = self.D.clone()
        # 流出速度向量（DVGO）
        self.OV = torch.zeros(num, dtype=torch.float32).to(self.device)

        # 当前位置向量（PPV）
        self.P = torch.zeros(num, dtype=torch.float32).to(self.device)
        self.P_old = self.P.clone()

        # 快速积分细胞
        self.A = torch.zeros(num, dtype=torch.float32).to(self.device)
        self.A_old = self.A.clone()
        # 慢速积分细胞
        self.B = torch.zeros(num, dtype=torch.float32).to(self.device)
        self.B_old = self.B.clone()
        # 调节信号向量
        self.R = torch.zeros(1, dtype=torch.float32).to(self.device)
        self.R_old = self.R.clone()

    def run(self, Si, Z):
        SZ = self.forward(Si, Z)
        self.update()
        return SZ

    def update(self):
        self.T_old = self.T.clone()
        self.D_old = self.D.clone()
        self.P_old = self.P.clone()
        self.A_old = self.A.clone()
        self.B_old = self.B.clone()
        self.R_old = self.R.clone()

    @torch.compile
    def forward(self, Si, Z):
        """
        :param Si: 目标位置向量
        :param Z:  意志信号 Z=0/1 当前为0(bottom-up) 回忆为1(top-down)
        """
        # G 对应论文中 GO，为意志门控信号
        self.G = self.dt * (-self.G_old + Z)  # [1]

        # 目标位置向量（TPV）
        self.T = self.T_old + self.dt * (-0.5 * self.T_old+(1-self.T_old) * (100 * (fun.relu(Si - 0.5))))  # [10]

        # D为差分向量（D）-> 目标位置向量（T）- 当前位置向量（P）
        self.D = self.D_old + self.dt * (-self.D_old + fun.relu(self.T_old) - self.P_old)  # [10]

        # 流出速度向量（DVGO）
        self.OV = fun.relu(self.D_old) * self.G_old  # [10]

        # 当前位置向量（PV）
        self.P = self.P_old + self.dt * self.OV  # [10]

        # 快速积分速度细胞
        self.A = self.dt * (3 * (-0.1 * self.A_old + self.G_old * torch.sum(fun.relu(self.D_old), dim=0)))  # [10]

        # 慢速积分速度细胞
        self.B = self.dt * (-0.1 * self.B_old + self.G_old * torch.sum(fun.relu(self.D_old), dim=0))  # [10]

        # 调节信号向量
        self.R = self.R_old + self.dt * (-self.R_old + Z + 10 * (self.B_old - self.A_old))  # [1]
        # print(self.R_old.shape)
        return self.P


if __name__ == '__main__':
    device = "cuda:0"
    dt = 0.1
    V = 0
    
    I_list = [[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0]]
    
    #I_list = [[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    I_list = torch.tensor(I_list).to(device)

    def process_plot(I_list):
        global k
        cognitive = Cognitive(0.1, 10, 2, "cuda:0")
        motor = Motor(10, 0.1, "cuda:0")
        vite = VITE(10, 0.1, "cuda:0")

        t = []
        X_history = [[] for _ in range(10)]  # 记录Xi的历史值
        Y_history = [[] for _ in range(10)]  # 记录Yi的历史值
        Cj_history = [[] for _ in range(6)]
        F_history = [[] for _ in range(10)]
        S_history = [[] for _ in range(10)]
        Q_history = [[] for _ in range(10)]
        G_history = [[] for _ in range(10)]
        T_history = [[] for _ in range(10)]
        D_history = [[] for _ in range(10)]
        P_history = [[] for _ in range(10)]
        A_history = [[] for _ in range(10)]
        B_history = [[] for _ in range(10)]
        B_A_history = [[] for _ in range(10)]   # 记录B-A的历史值

        K = torch.tensor(10)
        J = torch.tensor(10)
        time_step=0
        for i in range(len(I_list)):
            items=I_list[i]
            for _ in range(10):
                t.append(time_step)  # t即为I_list时刻
                Y = cognitive.run(items, K, J)
                S = motor.run(vite.G, vite.R, Y, Ei=torch.zeros(10).to(device))
                vite.run(S, V)   # V=0/1
                time_step+=dt
                # 记录各样本的值
                for k in range(10):
                    X_history[k].append(cognitive.Xi[k].cpu().numpy())
                    # print(np.array(X_history).shape)
                    Y_history[k].append(cognitive.Yi[k].cpu().numpy())
                    # print(np.array(Y_history).shape)
                    F_history[k].append(motor.Fi[k].cpu().numpy())
                    S_history[k].append(motor.Si[k].cpu().numpy())
                    Q_history[k].append(motor.Qi[k].cpu().numpy())
                    # print(np.array(Q_history))
                    # G_history[k].append(vite.G[k].cpu())
                    T_history[k].append(vite.T[k].cpu().numpy())
                    D_history[k].append(vite.D[k].cpu().numpy())
                    P_history[k].append(vite.P[k].cpu().numpy())
                    A_history[k].append(vite.A[k].cpu().numpy())
                    B_history[k].append(vite.B[k].cpu().numpy())
                    B_A_history[k].append(vite.A[k].cpu().numpy()-vite.B[k].cpu().numpy())
                for k in range(2):
                     Cj_history[k].append(cognitive.Cj[k].cpu().numpy())

            for _ in range(40):
                items=items*0
                t.append(time_step)  # t即为I_list时刻
                Y = cognitive.run(items, K, J)
                S = motor.run(vite.G, vite.R, Y, Ei=torch.zeros(10).to(device))
                vite.run(S, V)   # V=0/1
                time_step+=dt
                # 记录各样本的值
                for k in range(10):
                    X_history[k].append(cognitive.Xi[k].cpu().numpy())
                    # print(np.array(X_history).shape)
                    Y_history[k].append(cognitive.Yi[k].cpu().numpy())
                    # print(np.array(Y_history).shape)
                    F_history[k].append(motor.Fi[k].cpu().numpy())
                    S_history[k].append(motor.Si[k].cpu().numpy())
                    Q_history[k].append(motor.Qi[k].cpu().numpy())
                    # print(np.array(Q_history))
                    # G_history[k].append(vite.G[k].cpu())
                    T_history[k].append(vite.T[k].cpu().numpy())
                    D_history[k].append(vite.D[k].cpu().numpy())
                    P_history[k].append(vite.P[k].cpu().numpy())
                    A_history[k].append(vite.A[k].cpu().numpy())
                    B_history[k].append(vite.B[k].cpu().numpy())
                    B_A_history[k].append(vite.A[k].cpu().numpy()-vite.B[k].cpu().numpy())
                for k in range(2):
                     Cj_history[k].append(cognitive.Cj[k].cpu().numpy())

        color = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'magenta', 'brown']  # 10个线条用不同的颜色
        
        # 绘制Xi曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, X_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Xi Value')
        plt.grid(True)
        plt.savefig("./images/Xi.png")
            
        # 绘制Yi曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, Y_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Yi Value')
        plt.grid(True)
        plt.savefig("./images/Yi.png")

        # 绘制Cj曲线
        plt.clf() 
        for n in range(2):
            plt.plot(t, Cj_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Cj')
        plt.grid(True)
        plt.savefig("./images/C_J.png")

        # 绘制F曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, F_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('F Value')
        plt.grid(True)
        plt.savefig("./images/F.png")

        # 绘制S曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, S_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('S Value')
        plt.grid(True)
        plt.savefig("./images/S.png")

        # 绘制Q曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, Q_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/Q.png")

        # 绘制T曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, T_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/T.png")

        # 绘制D曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, D_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/D.png")

        # 绘制P曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, P_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/P.png")

        # 绘制A曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, A_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/A.png")

        # 绘制B曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, B_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/B.png")

        # 绘制B-A曲线
        plt.clf() 
        for n in range(10):
            plt.plot(t, B_A_history[n], linestyle='--', color=color[n])
        plt.xlabel('Time Step')
        plt.ylabel('Q Value')
        plt.grid(True)
        plt.savefig("./images/B_A.png")

    process_plot(I_list)








        





















