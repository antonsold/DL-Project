# DL-Project
This repository contains two course projects for the Deep Learning (EE-559) course in EPFL.
## First project
The goal was to test different architectures to compare two digits visible in a
two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an
auxiliary loss to help the training of the main objective.
## Second project
The objective of this project was to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules. The model was then trained and tested on a sample dataset of 2D points.

Each folder contains a `test.py` file to run without arguments and a PDF with the report.
