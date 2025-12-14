from TimeTaken import time_taken
@time_taken
def fun(a,b,c):
    for i in range(a):
        for j in range(b):
            for k in range(c):
                continue
    return a+b+c

if __name__ == '__main__':
    a = fun(2,3,4)
    