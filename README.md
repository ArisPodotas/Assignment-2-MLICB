# Code style

This section refers to my way of naming, using and implementing things in Python.

I like to write Python code as if I am writing a non interpreted language as far as function declaration variables go. Any segment that anyone other than me will have to look at use know will have type hints in accordance to the Python Typing [documentaion](https://docs.python.org/3/library/typing.html). I always include documentation string in my function however I won't be going into too much detail about the input parameters in the documentation strings since there is limited time. Variable names are reused within function and camel case is preferred. All arrays and lists are pre allocated to increase performance and only indexed into and assigned to new values. Tho ```del``` keyword, which many don't know/care about is used to clean memory when done with any variable and if a variable is kept it is likely due to reuse farther down the assignment.

The single responsibility principle is followed with the small caveat that there are 3 overloads every function could have

- Verbose
- Logging
- Plotting

These 3 overloads are self explanatory but their implementation is usually of checking a control flow statement and then running a distinct function within the overloaded one we are at.

# Links and References

All links to websites other than the github repository the assignment is implemented in will be of the url. All links of materials from within the github repository will use relative paths, should these relative paths not be supported in the in browser version of the assignment on github they should work once the repository has been cloned locally.

# Repository branches

It was suggested we use a Linux based machine for the assignment, I have two machines that I work with and one is Linux based the other is Windows based so the two branches are for the separate machines.

# Final deliverable

The "Report.pdf" is the report of the assignment

The bonus_1.tex, bonus_2.tex are unused and the reports for the bonus questions exist in the notebooks of the bonus questions themselves

# Fixes from the last assignment

I regret not making a saving function for the figures in the last assignment since I then went over the images in the notebooks to take screenshots of them and also as a results my report had figures added at the end making it hard to estimate the proper pages and setup wasting some time. This time I will implement saving functionality to all plotting functions so that I don't need to write all the figures at the end

My utils functions are now in one class that holds all the functions deemed as utilities.

