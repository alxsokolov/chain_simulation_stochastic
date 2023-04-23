import os

#clean up realisations
directory='dump'
test=os.listdir(directory)
for item in test:
    if item.endswith(".dump.npy"):
        os.remove(os.path.join(directory, item ))
