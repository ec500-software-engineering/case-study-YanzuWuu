Case studies on Scikit-learn
====
![logo](logo.png)

Technology and Platform used for development
----
What coding languages are used? 
Do you think the same languages would be used if the project was started today? 
What languages would you use for the project if starting it today?

What build system is used (e.g. Bazel, CMake, Meson)? 
What build tools / environment are needed to build (e.g. does it require Visual Studio or just GCC or ?)

What frameworks / libraries are used in the project? At least one of these projects don’t use any external libraries or explicit threading, yet is noted for being the fastest in its category--in that case, what intrinsic language techniques is it using to get this speed.

Answer:

a. Scikit-learn is a package written by python and also used on python. Some core algorithm are written in Cython. As a ML-package, comparing with tensorflow, Scikit-learn mainly focuses on help users process data by their own, such as selecting features, compressing dimensions, and converting formats. 

Python is the most popular language used in machine learning field, and as python’s exclusive package, if Scikit-learn is built today, it will also use python. Also, I’d like to use python too, because it has a lot of convenient third-party libraries of mathematical operations and Structured data manipulation, such as with NumPy, SciPy, Pandas.

b. Scikit-learn is optimized by different authors, and I didn’t find any resources showing the building system of Scikit-learn.

c. Scikit-learn’s algorithm library is built on top of SciPy (an open source Python-based scientific computing toolkit) - you must install SciPy before you can use SciKit-learn. And also need the library NumPy.

Testing: describe unit/integration/module tests and the test framework
----
How are they ensuring the testing is meaningful? Do they have code coverage metrics for example?

What CI platform(s) are they using (e.g. Travis-CI, AppVeyor)?

What computing platform combinations are tested on their CI? E.g. Windows 10, Cygwin, Linux, Mac, GCC, Clang

Answer:

a. They use codecov to test the code coverage metrics.
![codecov1](codecov1.png)
![codecov2](codecov2.png)

b.  They use Travis-CI to test the units.
![travis](travis.png)

c.  Linux environment is tested on Travis-CI.
![linux_enviro](linux_enviro.png)

Software architecture in your own words
-----
Including:How would you add / edit functionality if you wanted to? How would one use this project from external projects, or is it only usable as a standalone program?

What parts of the software are asynchronous (if any)?

Please make diagrams as appropriate for your explanation

How are separation of concerns and information hiding handled?

What architectural patterns are used？

Does the project lean more towards object oriented or functional components

Answer:

a.  Before wanting to add a new algorithm, which is usually a major and lengthy undertaking, we ought to start with “Know issue” on their github, in case some change may already have solutions. And they have some issue tags for the beginner to get familiar with contribution. Some “help wanted” issues also need contributors to solve.

Any sort of documentation is ok if we want to make some change, and we have to generate a HTML output by typing make html from the doc/ directory.When you change the documentation in a pull request, CircleCI automatically builds it. Also, use pytest package to test the units.

Users can use Scikit-learn as a data process tool, which means it can be used as many single part or functions.

b. Emm, we use it only when we call some specific functions.

c. The diagram below is a classic diagram of scikit-learn structure of different function branches.
![sklearn_diagram](sklearn_diagram.png)


d. In the github, they make a great readme and seperate different part in different folder. And in the user guide, they give a lot of link from one question to another possible question you may come out, which is very clearly. (if this is the point of the problem d)

e. Actually I’m not sure if the Scikit-learn has a “architectural pattern”, because “architectural pattern” is a concept used in software.

f. Yes, the Scikit-learn can be divided into six components: Classification, regression, clustering, data dimensionality reduction, model selection and data preprocessing. And every algorithm is packaged well in functions and the modules in scikit-learn are highly abstracted, increases the efficiency of the model, reducing the difficulty of batching and standardization (by using pipeline).



 Analyze two defects in the project
 -------
 --e.g. open GitHub issue, support request tickets or feature request for the project
 
 Does the issue require an architecture change, or is it just adding a new function or?
 
 make a patch / pull request for the project to fix problem / add feature
 
a. It depends on the issue type, I think most issues don’t have to change the architecture change.

b. It’s hard to give a change or a optimization by yourself, because you have to communicate with others in the issue page on github first and may find someone to correct the bug or do some optimization. The team is fixed by the way. And one good side is you can train yourself from some easy issues proposed by others and correct them. 

 Making a demonstration application of the system, your own application showing how the software is used
 -------
 
I used Scikit-learn to do some data preprocessing （standardization）and seperate data, then do the logistic regression and create a classification report of prediction.

The code is updated on the github.

And the result is as followed:
![result](result.png)


Reference:
------
Scikit-learn github:
https://github.com/scikit-learn/scikit-learn

Scikit-learn user guide:
https://scikit-learn.org/0.20/_downloads/scikit-learn-docs.pdf

An introduction to machine learning with scikit-learn:
https://scikit-learn.org/stable/tutorial/basic/tutorial.html

Introduction of Scikit-learn:
https://en.wikipedia.org/wiki/Scikit-learn

