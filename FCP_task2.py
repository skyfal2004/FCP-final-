import random
import numpy as np
import matplotlib.pyplot as plt
import sys
def defuant_main(Threshold = 0.2, Coupling = 0.2, testTimes = 100):
    array = np.random.uniform(low = 0,high = 1,size=100)
    array_2d = []
    for i in range(testTimes*100):
        position_random_element = random.randrange(len(array))
        if position_random_element == 0:
            random_element_neighbor = position_random_element + 1
            # print('the random element is the first number and it has no left side neighbor')
        elif position_random_element == len(array) -1:
            random_element_neighbor= position_random_element - 1
            # print('the random element is the final number and it has no right side neighbor')
        else:
            random_element_neighbor = position_random_element +random.choice([1,-1])
        if abs(array[random_element_neighbor] - array[position_random_element]) < Threshold:
            temp = array[position_random_element]
            array[position_random_element] = array[position_random_element] +Coupling*(array[random_element_neighbor] - array[position_random_element])
            array[random_element_neighbor] = array[random_element_neighbor] + Coupling*(temp - array[random_element_neighbor])
        if i % 100 == 0:
            array_2d.append(array.copy())
        # elif abs(array[random_element_neighbor] - array[random_element]) > Threshold:
        #     print('the two selected people have a difference of opinion greater than a threshold')



    rows = len(array_2d)
    cols = len(array_2d[0])



    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,5))
    ax1.hist(array, bins=10, range = (0.0, 1.0), edgecolor='black')

    ax1.set_xlabel('opinion')
    ax1.set_ylabel('time')
    ax1.set_xlim(0, 1)
    rows = len(array_2d)
    cols = len(array_2d[0])
    for row in range(rows):
        plt.scatter([row] * cols, array_2d[row], c='red')
    ax2.set_ylim(0.0,1.0)
    ax2.set_xlim(1,testTimes)
    ax2.set_xlabel('Times')
    ax2.set_ylabel('Opinion')
    plt.suptitle('Coupling ='+ str(Coupling)+ ', Threshold =' +str(Threshold))
    plt.show()
    pass



def test_defuant():
    defuant_main(Coupling=0.5,Threshold=0.5)
    defuant_main(Coupling=0.5,Threshold=0.1)
    defuant_main(Coupling=0.1,Threshold=0.5)
    defuant_main(Coupling=0.1,Threshold=0.1)
    pass


def main():
    arg = sys.argv
    if arg[1] == "-defuant":
        if len(arg) == 2:
            defuant_main(Threshold=0.2, Coupling=0.2)
        elif len(arg) == 4:
            if arg[2] == "-beta":
                defuant_main(Threshold=0.2, Coupling=float(arg[3]))
            if arg[2] == "-threshold":
                defuant_main(Threshold=float(arg[3]), Coupling=0.2)
        elif len(arg) == 6:
            defuant_main(Threshold=float(arg[arg.index("-threshold") + 1]), Coupling=float(arg[arg.index("-beta") + 1]))
    if arg[1] == "-test_defuant":
        test_defuant()
    pass

if __name__ == '__main__':
    main()