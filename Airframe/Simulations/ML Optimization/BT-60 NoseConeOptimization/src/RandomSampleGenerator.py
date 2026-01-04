import random
filepath = input("Filepath: ")
n = int(input("Number of Samples: "))

with open(filepath+"/dataset.csv", "a") as file:
    file.write("\n")
    for i in range(n):
        #Non Sorted: file.write(",".join([str(random.uniform(0,1)) for j in range(4)])+",\n")
        #Sorted: 
        scales=[9,9,1.65]
        file.write(",".join(map(str, [scales[j]*random.uniform(0,1) for j in range(3)]))+",\n")
    file.close()
print("Done")