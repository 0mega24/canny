# canny
Canny Edge Detection in Python CuPy with Cuda C kernels

## I first take the original image:

![original](./Images/step0_original.png)

## I then make it grayscale

![gray](./Images/step1_gray.png)

## I then blur the image to reduce noise and make the edge detection clearer:

![blur](./Images/step2_blurred.png)

## I then get the x y and square root combined gradients:

![gradient x](./Images/step3_grad_x.png)
![gradient y](./Images/step3_grad_y.png)
![combined gradient](./Images/step3_grad_mag.png)

## I then do NMS on the image to thin down the edges:

![nms](./Images/step4_nms.png)

If I had had more time I would have then done double threshold and hysteresis to further reduce noise within the lines and to make them more complete paths and less disjointed.

## This leaves me with this output currently:

![final](./Images/output.png)

## CU messing about

As you can see in the repo there is also a cu file that does nearly the exact same thing with nearly the same CUDA kernels as well, however due to issues I was having memory allocation and it causing my laptop to crash.
