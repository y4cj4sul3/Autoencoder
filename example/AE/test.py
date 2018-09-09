import os

filepath = os.path.dirname(os.path.abspath(__file__))

cedpath = os.getcwd()

joinedpath = os.path.join(filepath, 'Model/', 'test/', 'ooo')


print(filepath)
print(cedpath)
print(joinedpath)


