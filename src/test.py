import Testing.create_test as ct
import sys


def main():
    param1 = int(sys.argv[1])
    param2 = int(sys.argv[2])
    F=ct.TestCreator()
    for i in range(param1,param2+1):
        F.generate_test(i)


if __name__ == "__main__":  
    main()