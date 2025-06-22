import matplotlib.pyplot  as plt

def GetLoss(path):
    loss = []
    x = 0
    num = 0
    with open(path, "r") as fp:
        for line in fp.readlines():
            if "Loss:" in line and "Time:" in line:
                x += float(line.split(' ')[-1])
                num += 1
            if "[====================]" in line:
                x /= num
                loss.append(x)
                x = 0
                num = 0
    return loss

def DrawLoss(loss):
    x = [i for i in range(len(loss))]
    y = loss
    plt.figure(1)
    plt.plot(x, y, label="Training Loss")
    plt.title('LeNet5-MNIST loss')
    plt.legend()
    plt.savefig("./log/loss.png", dpi=600)

if __name__ == "__main__":
    loss = GetLoss("./log/lumos_t.log")
    DrawLoss(loss)