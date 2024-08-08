import numpy as np
import matplotlib.pyplot as plt

class Lorenz():
    def __init__(self, xx, yy, zz, dt):
        self.sigma = 10
        self.beta = 8/3
        self.rho = 28
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self. dt = dt
    
    def calculate(self):
        for i in range(1, 50000):
            x = self.xx[-1]
            y = self.yy[-1]
            z = self.zz[-1]
            self.xx.append(x + self.dt * self.sigma*(y - x))
            self.yy.append(y + self.dt * (x * (self.rho - z) - y))
            self.zz.append(z + self.dt * (x * y - self.beta * z))
        
        output = np.array([self.xx, self.yy, self.zz]).T
        print('Data Shape:',output.shape)
        input = np.zeros((len(output),1))
        print('Input Shape:',input.shape)
        t = np.arange(len(output)) * self.dt
        return input, output, t
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.xx, self.yy, self.zz, lw=.1, color='purple')
        ax.set_title('Lorenz Dynamics')
        plt.show()


lorenz = Lorenz([.1], [.1], [.1], 0.01)
input, ouput, t = lorenz.calculate()
lorenz.plot()