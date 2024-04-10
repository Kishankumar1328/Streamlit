import numpy as np
import streamlit as st

st.title("Numpy Tutorial")

pages = ["numpy", "arr", "ca", "Ip", "io", "dtype","aam","am","c","af","copy","sa","ssi","arrm"]
current_page = st.session_state.get("current_page", 0)

def navigate_to(page_number):
    st.session_state["current_page"] = page_number

def numpy():
    st.header("NUMPY")

    st.subheader("The NumPy library is the core library for scientific computing inPython."
             " It provides a high-performance multidimensional array"
             "object, and tools for working with these arrays")
    st.button("Next", on_click=lambda: navigate_to(1))

def arr():
    st.header("Numpy Array")

    image1='numpy.png'
    st.image(image1,caption="1 Dimensional Array")
    st.button("Previous", on_click=lambda: navigate_to(0))
    st.button("Next", on_click=lambda: navigate_to(2))


def ca():
    st.header("Creating Arrays")
    st.markdown("---")

    st.latex("1 Dimensional Array")
    st.code("np.array([1,2,3])")
    st.markdown("---")

    st.latex("2 Dimensional Array")
    st.code("np.array([[1,3,2,9],[0,1,0,5]],dtype=float32)")
    st.markdown("---")

    st.latex("3 Dimensional Array")
    st.code("np.array([[[1,3],[2,5],[2,8],[2,9]]],dtype=int64)")
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(1))
    st.button("Next", on_click=lambda: navigate_to(3))

def Ip():
    st.header("Initial Placeholders")
    st.markdown("---")

    st.subheader("Create an array of zeros")
    st.code("np.zeros((4,4))")
    st.markdown("---")

    st.subheader("Create an array of ones")
    st.code("np.ones((2,3,4),dtype=np.float)")
    st.markdown("---")

    st.subheader("Create an array of evenly")
    st.code("np.arange(10,25,5)")
    st.markdown("---")

    st.subheader("Create an array of evenly spaced values")
    st.code("np.linspace(0,2,9) ")
    st.markdown("---")

    st.subheader("Create a constant array")
    st.code("np.full((2,2),7) ")
    st.markdown("---")

    st.subheader("Create a 2X2 identity matrix")
    st.code("np.eye(2) ")
    st.markdown("---")

    st.subheader("Create an array with random values")
    st.code("np.random.random((2,2)) ")
    st.markdown("---")

    st.subheader("Create an empty array")
    st.code("np.empty((3,2))")
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(2))
    st.button("Next", on_click=lambda: navigate_to(4))

def io():
    st.header("I/O")
    st.markdown("---")

    st.subheader("Saving & Loading On Disk")
    st.code("np.save('my_array', a)")
    st.code("np.savez('array.npz', a, b)")
    st.code(" np.load('my_array.npy')")
    st.markdown("---")

    st.subheader("Saving & Loading Text Files")
    st.code("np.loadtxt('myfile.txt')")
    st.code("np.genfromtxt('my_file.csv', delimiter=',')")
    st.code("np.savetxt('myarray.txt', a, delimiter='')")
    st.markdown("---")
    st.button("Previous", on_click=lambda: navigate_to(3))
    st.button("Next", on_click=lambda: navigate_to(5))

def dtypes():
    st.header("Data Types")
    st.markdown("---")

    st.subheader("Signed 64-bit integer types")
    st.code("np.int64")
    st.markdown("---")

    st.subheader("Standard double-precision floating point")
    st.code("np.float32")
    st.markdown("---")

    st.subheader("Complex numbers represented by 128 floats")
    st.code("np.complex")
    st.markdown("---")

    st.subheader("Boolean type storing TRUE and FALSE values")
    st.code("np.bool")
    st.markdown("---")

    st.subheader("Python object type")
    st.code("np.object")
    st.markdown("---")

    st.subheader("Fixed-length string type")
    st.code("np.string_ ")
    st.markdown("---")

    st.subheader("Fixed-length unicode type")
    st.code("np.unicode_")
    st.markdown("---")
    st.button("Previous", on_click=lambda: navigate_to(4))
    st.button("Next", on_click=lambda: navigate_to(6))

def aam():
    st.header("Array Attributes and Methods")
    st.markdown("---")

    st.subheader("Array dimensions")
    st.code("a.shape")
    st.markdown("---")

    st.subheader("Length of array")
    st.code("len(a)")
    st.markdown("---")

    st.subheader("Number of array dimensions")
    st.code("a.ndim")
    st.markdown("---")

    st.subheader("Number of array elements")
    st.code("a.size")
    st.markdown("---")

    st.subheader("Data type of array elements")
    st.code("a.dtype")
    st.markdown("---")

    st.subheader("Name of data type")
    st.code("a.dtype.name")
    st.markdown("---")

    st.subheader("Convert an array to a different type")
    st.code("b.astype(int) ")
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(5))
    st.button("Next", on_click=lambda: navigate_to(7))



def am():
    st.header("Arithmetic Operations")

    st.subheader("Addition")
    ad="""
import numpy as np
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[3,2,1]])
result=arr1+arr2
print(result)"""

    adb="""
import numpy as np
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[3,2,1]])
result=np.add(arr1,arr2)
print(result)
"""
    st.code(ad)
    st.write("Or")
    st.code(adb)
    st.markdown("---")

    st.subheader("Subtraction")
    sub1 = """
    import numpy as np
    arr1=np.array([[1,2,3],[4,5,6]])
    arr2=np.array([[7,8,9],[10,11,12]])
    result=arr1-arr2
    print(result)"""

    sub2 = """
    import numpy as np
    arr1=np.array([[13,29,31],[32,25,28]])
    arr2=np.array([[5,1,8],[12,2,4]])
    result=np.subtract(arr1,arr2)
    print(result)
    """
    st.code(sub1)
    st.write("Or")
    st.code(sub2)
    st.markdown("---")

    st.subheader("Multiplication")
    mult1= """
    import numpy as np
    arr1=np.array([[1,2,3],[4,5,6]])
    arr2=np.array([[7,8,9],[3,2,1]])
    result=arr1*arr2
    print(result)
    """
    mult2 = """
     import numpy as np
     arr1=np.array([[1,2,3],[4,5,6]])
     arr2=np.array([[7,8,9],[3,2,1]])
     result=np.multiply(arr1,arr2)
     print(result)
     """
    st.code(mult1)
    st.write("Or")
    st.code(mult2)
    st.markdown("---")

    st.subheader("Divide")
    div1= """
    import numpy as np
    arr1=np.array([[1,2,3],[4,5,6]])
    arr2=np.array([[7,8,9],[3,2,1]])
    result=arr1/arr2
    print(result)"""

    div2 = """
    import numpy as np
    arr1=np.array([[1,2,3],[4,5,6]])
    arr2=np.array([[7,8,9],[3,2,1]])
    result=np.divide(arr1,arr2)
    print(result)
    """
    st.code(div1)
    st.write("Or")
    st.code(div2)
    st.markdown("---")

    st.subheader("Exponentiation")
    st.code("np.exp(a)")
    st.markdown("---")

    st.subheader("Square Root")
    st.code("np.sqrt(a)")
    st.markdown("---")

    st.subheader("sines of an array")
    st.code("np.sin(a)")
    st.markdown("---")

    st.subheader("cosine of an array")
    st.code("np.cos(a)")
    st.markdown("---")

    st.subheader("Element-wise natural logarithm")
    st.code("np.log(a)")
    st.markdown("---")

    st.subheader("Dot Product")
    dp="""
import numpy as np
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
dot_product = np.dot(array1, array2)
print(dot_product)
"""

    st.code(dp)
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(6))
    st.button("Next", on_click=lambda: navigate_to(8))


def c():
    st.header("Comparsion")


    cta="""
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[1, 3, 3],
              [7, 5, 8]])
element_wise_comparison = a == b
print("Element-wise comparison:")
print(element_wise_comparison)

# Element-wise comparison with a scalar
element_wise_comparison_scalar = a < 2
print("\nElement-wise comparison with a scalar:")
print(element_wise_comparison_scalar)

# Array-wise comparison
array_wise_comparison = np.array_equal(a, b)
print("\nArray-wise comparison:")
print(array_wise_comparison)

"""
    st.code(cta)
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(7))
    st.button("Next", on_click=lambda: navigate_to(9))


def af():
    st.header("Aggregate Functions")

    st.subheader("Array wise sum")
    st.code("a.sum()")
    st.markdown("---")

    st.subheader("Array-wise minimum value")
    st.code("a.min()")
    st.markdown("---")

    st.subheader("Maximum value of an array row")
    st.code(" b.max(axis=0)")
    st.markdown("---")

    st.subheader("Cumulative sum of the elements")
    st.code("b.cumsum(axis=1)")
    st.markdown("---")

    st.subheader("Mean")
    st.code("a.mean()")
    st.markdown("---")

    st.subheader("Median")
    st.code("a.median()")
    st.markdown("---")

    st.subheader("Correlation coefficient")
    st.code("a.corrcoef()")
    st.markdown("---")

    st.subheader("Standard Deviation")
    st.code("np.std(a)")
    st.markdown("---")

    aff="""
    import numpy as np

# Define arrays
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[7, 8, 9],
              [10, 11, 12]])

# Array-wise sum
array_sum = a.sum()
print("Array-wise sum:", array_sum)

# Array-wise minimum value
array_min = a.min()
print("Array-wise minimum value:", array_min)

# Maximum value of each column
max_per_column = b.max(axis=0)
print("Maximum value of each column:", max_per_column)

# Cumulative sum of the elements along each row
cumulative_sum = b.cumsum(axis=1)
print("Cumulative sum of the elements along each row:")
print(cumulative_sum)

# Mean of all elements
array_mean = a.mean()
print("Mean of all elements:", array_mean)

# Standard deviation
array_std = np.std(b)
print("Standard deviation:", array_std)
"""
    st.code(aff)
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(8))
    st.button("Next", on_click=lambda: navigate_to(10))


def copy():
    st.header("Copying Arrays")

    st.subheader("Create a view of the array with the same data")
    st.code("a.view()")
    st.markdown("---")

    st.subheader("Create a copy of the array")
    st.code("np.copy(a)")
    st.markdown("---")

    st.subheader("Create a deep copy of the array")
    st.code("a.copy()")
    st.markdown("---")

    cpp="""
import numpy as np

# Define an array
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Create a view of the array with the same data
h = a.view()
print("View of the array with the same data:")
print(h)

# Create a copy of the array
copy_a = np.copy(a)
print("\nCopy of the array:")
print(copy_a)

# Alternatively, you can also use:
h = a.copy()
print("\nAnother way to create a copy of the array:")
print(h)
"""
    st.code(cpp)
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(9))
    st.button("Next", on_click=lambda: navigate_to(11))


def sa():

    st.header("Sorting Arrays")


    st.subheader("Sort an array")
    st.code("a.sort()")
    st.markdown("---")

    st.subheader("Sort the elements of an array's axis")
    st.code("a.sort(axis=0)")
    st.markdown("---")

    py="""
    
import numpy as np

# Define arrays
a = np.array([[3, 2, 1],
              [6, 5, 4]])

c = np.array([[9, 8, 7],
              [12, 11, 10]])

# Sort array 'a'
a.sort()
print("Sorted array 'a':")
print(a)

# Sort along the specified axis for array 'c'
c.sort(axis=0)
print("\nSorted array 'c' along axis 0:")
print(c)
"""
    st.code(py)
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(10))
    st.button("Next", on_click=lambda: navigate_to(12))

def ssi():

    st.header("Subsetting, Slicing, Indexing")
    st.markdown("---")

    st.subheader("Subsetting")
    st.code("a[3]")
    st.write("Select the element at the 2nd index")
    st.code("b[1,3]")
    st.write("Select the element at row 1 column 2 (equivalent to b[1][2])")
    st.markdown("---")

    st.write("code")
    kk="""
import numpy as np

# Define arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Select the element at the 2nd index of array 'a'
element_a = a[2]
print("Element at the 2nd index of array 'a':", element_a)

# Select the element at row 1, column 2 of array 'b'
element_b = b[1, 2]
print("Element at row 1, column 2 of array 'b':", element_b)
"""
    st.code(kk)
    st.markdown("---")

    st.subheader("Slicing")
    st.markdown("---")
    st.code("a[0:2] ")
    st.write("Select items at index 0 and 1")
    st.markdown("---")

    st.code("b[0:2,1]")
    st.write("Select items at rows 0 and 1 in column 1")
    st.markdown("---")

    st.code("a[ : :-1] ")
    st.write("Reversed array ")
    st.markdown("---")

    st.subheader("Boolean Indexing")
    st.code("a[a<2]")
    st.write("Select elements from a less than 2")
    st.markdown("---")

    st.button("Previous", on_click=lambda: navigate_to(11))
    st.button("Next", on_click=lambda: navigate_to(13))

def arrm():
    st.header("Array Manipulation")
    st.markdown("---")

    st.subheader("Transposing Array")
    tk="""
import numpy as np
i=np.transpose(b)
i.T"""
    st.code(tk)
    st.markdown("---")

    st.subheader("Changing Array Shape")

    st.code("b.ravel()")
    st.write("Flatten the array")
    st.markdown("---")

    st.code("g.reshape(3,-2)")
    st.write("Reshape, but don’t change data")
    st.markdown("---")

    skk="""
import numpy as np

# Define arrays
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

g = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Flatten the array 'b'
flattened_b = b.ravel()
print("Flattened array 'b':", flattened_b)

# Reshape the array 'g' to have 3 rows and as many columns as needed
reshaped_g = g.reshape(3, -2)
print("Reshaped array 'g' with 3 rows and as many columns as needed:")
print(reshaped_g)
"""
    st.code(skk)

    st.subheader("Adding/Removing Elements")
    st.markdown("---")
    st.code("np.concatenate((a,b),axis=0) ")
    st.write("Concatenate arrays")
    st.markdown("---")

    st.code("np.vstack((a,b))")
    st.write("Stack arrays vertically (row-wise)")
    st.markdown("---")

    st.code("np.r_[a,b] ")
    st.write("Stack arrays vertically (row-wise)")
    st.markdown("---")

    st.code("np.hstack((a,b)) ")
    st.write("Stack arrays horizontally (column-wise)")
    st.markdown("---")

    st.code("np.column_stack((a,b))")
    st.write("Create stacked column-wise arrays")
    st.markdown("---")

    kkk="""
import numpy as np

# Define arrays
a = np.array([1, 2, 3])
b = np.array([[10, 15, 20],
              [1, 0, 1]])
d = np.array([[7, 7],
              [7, 7]])

# Concatenate arrays 'a' and 'b' along axis 0
concatenated_ab_axis0 = np.concatenate((a, b), axis=0)
print("Concatenated arrays 'a' and 'b' along axis 0:")
print(concatenated_ab_axis0)

# Stack arrays 'a' and 'b' vertically (row-wise)
stacked_ab_vertically = np.vstack((a, b))
print("\nStacked arrays 'a' and 'b' vertically (row-wise):")
print(stacked_ab_vertically)

# Stack arrays 'a' and 'b' vertically (row-wise) using np.r_
stacked_ab_vertically_r = np.r_[a, b]
print("\nStacked arrays 'a' and 'b' vertically (row-wise) using np.r_:")
print(stacked_ab_vertically_r)

# Stack arrays 'a' and 'b' horizontally (column-wise)
stacked_ab_horizontally = np.hstack((a[:, np.newaxis], b))
print("\nStacked arrays 'a' and 'b' horizontally (column-wise):")
print(stacked_ab_horizontally)

# Create stacked column-wise arrays using np.column_stack()
stacked_column_wise = np.column_stack((a, d))
print("\nStacked column-wise arrays using np.column_stack():")
print(stacked_column_wise)

# Stack arrays 'a' and 'd' column-wise using np.c_
stacked_column_wise_c = np.c_[a, d]
print("\nStacked arrays 'a' and 'd' column-wise using np.c_:")
print(stacked_column_wise_c)
"""
    st.code(kkk)


    st.subheader("Splitting Arrays")
    st.markdown("---")
    st.code("np.hsplit(a,3)")
    st.write("Split the array horizontally at the 3rd index")
    st.markdown("---")

    st.code("np.vsplit(b,3)")
    st.write("Split the array vertically at the 2nd index")
    st.markdown("---")

    srk="""
import numpy as np

# Define arrays
a = np.array([1, 2, 3])
b = np.array([[[1.5, 2., 1.],
               [4., 5., 6.]],
              [[3., 2., 3.],
               [4., 5., 6.]],
              [[7., 8., 9.],
               [10., 11., 12.]]])

# Split the array 'a' horizontally at the 3rd index
hsplit_a = np.hsplit(a, 3)
print("Split array 'a' horizontally at the 3rd index:")
print(hsplit_a)

# Split the array 'b' vertically at the 2nd index
vsplit_b = np.vsplit(b, 2)
print("\nSplit array 'b' vertically at the 2nd index:")
print(vsplit_b)
"""
    st.code(srk)
    st.markdown("---")

    
















if current_page == 0:
    numpy()
elif current_page == 1:
    arr()
elif current_page == 2:
    ca()
elif current_page == 3:
    Ip()
elif current_page == 4:
    io()
elif current_page == 5:
    dtypes()
elif current_page == 6:
    aam()
elif current_page == 7:
    am()
elif current_page ==8:
    c()
elif current_page ==9:
    af()
elif current_page ==10:
    copy()
elif current_page ==11:
    sa()
elif current_page ==12:
    ssi()
elif current_page ==13:
    arrm()

