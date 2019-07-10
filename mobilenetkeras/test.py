import numpy as np
import pandas as pd
import csv
import json

history = {'val_loss': [0.8511574417352676, 0.7135399873942545, 0.4959253613671211], 'val_acc': [0.6374999992549419, 0.726027397260274, 0.780821922707231], 'loss': [0.9612156337415668, 0.7002647794973074, 0.6010482326445431], 'acc': [0.5676056393225428, 0.7160056661022959, 0.7584985808255652]}

# print(history['val_loss'])
result = zip(history['val_loss'],history['val_acc'],history['loss'],history['acc'])
result = set(result)
# print(result)

a = [list(a) for a in zip(history['val_loss'],history['val_acc'],history['loss'],history['acc'])]
print(a)

# for a in a:
#     print(a)
# rows = json.loads(history)
# r=zip(*rows.values())

# for key,value in history.items():
#     print(key)
#     print(value[0])
#     for data in zip(value):
#         print(data)

# print(history['val_loss'])
csv.register_dialect('myDialect', delimiter=',', quoting=csv.QUOTE_NONE)

csv_file = "history.csv"
# with open(csv_file, 'w') as csvfile:
#     writer = csv.writer(csvfile,dialect='myDialect')
#     # for i in range (len(a[0])):
#     writer.writerows(a)
# csvfile.close()

np.savetxt(csv_file, a,delimiter=',',header= "val_loss, val_acc,loss,acc")

my_array = np.loadtxt('history.csv',delimiter=",", skiprows=1)
print(my_array)