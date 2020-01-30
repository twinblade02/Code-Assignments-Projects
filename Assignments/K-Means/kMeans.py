print("DATA-51100", "Spring 2020")
print("Lionel Dsilva")
print("Statistical Programming - Assignment 2")

inpFile=input("Enter the name of the input file: ")
outFile=input("Enter the name of the output file: ")
numCluster=input("Enter the number of clusters: ")
k=int(numCluster)
points=[]
# Open input file
with open(inpFile) as fp:
    # Read each line in the file
    lin=fp.readline()
    while lin:
        points.append(float(lin))
        lin=fp.readline()
clus=[]
clusPoints=[]
previous=[]
current=[]
# Initialise cluster
for i in range(0,k):
    ls=[]
    lst=[]
    ls.append(points[i])
    clus.append(points[i])
    clusPoints.append(lst)
numberOfPoints=len(points)

distance=[]
it=1
while(True):
    # For empty list
    for x in range(0,k):
        clusPoints[x]=[]
    # Calculate the euclidean distance
    for i in range(0,numberOfPoints):
        for j in range(0,k):
            point=clus[j]
            distance.append(abs(points[i]-point))
        # Find minimum distance
        minValue=min(distance)
        index=distance.index(minValue)
        ls=clusPoints[index]
        ls.append(points[i])
        distance=[]
    # Print output
    print("Iteration "+str(it))
        
    # Print the point
    for y in range(0,k):
        print(str(y),end=' ')
        print(clusPoints[y])
    current=[]
    for x in range(0,k):
        current.append(clusPoints[x])
            
    # If current list and previous list are same; break
    if current==previous:
            break
            
    # Find new centroid
    for i in range(0,k):
        ls=clusPoints[i]
        numPoints=len(ls)
        total=0.0
            
    # Average of points in that cluster
        for j in range(0,numPoints):
            total=total+ls[j]
            avg=total/numPoints
            clus[i]=avg
    previous=[]
    for x in range(0,k):
        previous.append(clusPoints[x])
    it=it+1

# Open the output file in write mode
f= open(outFile,"w+")
    
# Write each point
for i in range(0,numberOfPoints):
    for j in range(0,k):
        if points[i] in clusPoints[j]:
            f.write("Point "+str(points[i])+" in cluster "+str(j)+"\n")
# Close files
f.close()
fp.close()