
def plotClusters(websites, clusters=8,xFactor=10, yFactor=10, myDpi=96, sampleSize=20, imagePath=None):
    """
     We want to plot every image according to the appropriate point
     on the x-axis according to its cluster number. We want to plot
     each new member of a given cluster at a higher y position

     Inputs:
         imagePath is a string to where the images are located
         websites is a list of tuples of the form:
             (clusterNumber, websitename)
    """
    if not imagePath:
        imagePath = getDataDir('top-15k', cut=True, big=False)
    clusterDict = [0 for n in xrange(clusters)]
    plt.figure(figsize=(800/myDpi, 800/myDpi), dpi=myDpi)
    for site in websites:
        clusterIndex, address = site
        try:
            yIndex = clusterDict[clusterIndex]
            print "yindex is : " + str(yIndex)
            print "sampleSize is: " + str(sampleSize)
            if yIndex > sampleSize:
                continue
            print imagePath+address
            image = mpimg.imread(imagePath+address)
            y = yIndex * yFactor
            clusterDict[clusterIndex]+=1
            plt.figimage(image, clusterIndex*xFactor, y)
        except IOError:
            # usually if we don't have the small cut image yet
            pass

    plt.show()

def readClustersAsDict(filename):
    with open(filename, 'r') as csvfile:
        current = 0
        clusters = {}
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            cluster = int(row[0])
            rank = int(row[1])
            clusters[rank] = cluster
        return clusters

def readClusters(filename):
    with open(filename, 'r') as csvfile:
        current = 0
        clusters = [[]]
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            cluster = int(row[0])
            if current < cluster:
                assert(current == cluster - 1)
                current += 1
                clusters.append([])
            assert(current == cluster)
            clusters[current].append((int(row[1]), row[2]))
        return clusters
