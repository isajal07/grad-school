import numpy as np
import data_storage as ds
import decision_trees as dt

file_name = "cat_dog_data.csv"

data = np.genfromtxt(file_name, dtype=str, delimiter=',')

a_samples,a_labels = ds.build_nparray(data)
print(type(a_labels))
print(a_samples)
print(a_labels,"\n")

l_samples,l_labels = ds.build_list(data)
print(type(l_labels))
for row in l_samples:
  print(row)
print(l_labels,"\n")

d_samples, d_labels = ds.build_dict(data)
print(type(d_labels))
for index in range(len(d_samples)):
    print(d_samples[index])
print(d_labels,"\n")

# Test for Binary values
max_depth = 3
DT = dt.DT_train_binary(a_samples,a_labels,max_depth)
test_acc = dt.DT_test_binary(a_samples,a_labels,DT)
print("DT:",test_acc)


# Test for real values
file_name_real = "data_2.csv"

data_real = np.genfromtxt(file_name_real, dtype=str, delimiter=',')

a_samples_real,a_labels_real = ds.build_nparray(data_real)
print(type(a_samples_real))
print(a_samples_real)
print(a_labels_real,"\n")

max_depth_real = 3
DT = dt.DT_train_real(a_samples_real,a_labels_real,max_depth_real)
test_acc = dt.DT_test_real(a_samples_real,a_labels_real,DT)
print("DT:",test_acc)