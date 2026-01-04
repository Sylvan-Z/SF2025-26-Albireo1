import random
filepath = input("Filepath: ")
n = int(input("Number of Samples: "))

with open(filepath+"/dataset.csv", "a") as file:
    file.write("\n")
    for i in range(n):
        #Non Sorted: file.write(",".join([str(random.uniform(0,1)) for j in range(4)])+",\n")
        #Sorted: 
        file.write(",".join(map(str, sorted([random.uniform(0,1) for j in range(4)])))+",\n")
    file.close()
print("Done")