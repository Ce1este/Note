# Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. 

Now we need to estimate the parameters in the hypothesis function. 

That's where **gradient descent** comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$	
(actually we are graphing the cost function as a function of the parameter estimates). 
  
We are not graphing x and y itself, but the **parameter** range of our **hypothesis function** and the **cost** resulting from selecting a particular set of parameters.

We put $\theta_0$ on the x axis and $\theta_1$ on the y axis, 
with the cost function on the vertical z axis. 

The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. 
The graph below depicts such a setup.

![img](../img/J(θ)-2%20parameters.png)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. 
when its value is the minimum. 

The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative(取导数) (the tangential line to a function) of our cost function. 

The slope of the tangent is the derivative at that point(切线的斜率是该点的导数) 
and it will give us a direction to move towards. 

We make steps down the cost function in the direction with the steepest descent. 

The size of each step is determined by the **parameter α**, which is called **the learning rate**.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α.

A smaller α would result in a smaller step and a larger α results in a larger step. 

The direction in which the step is taken is determined by the partial derivative(偏导数 of $J(\theta_0,\theta_1)$. 

Depending on where one starts on the graph, one could end up at different points. 

The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

**repeat until convergence(收敛):**

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$ 

where

$j=0,1$ represents the feature index number.

At each iteration(迭代) j, one should **simultaneously update** the parameters $\theta_1, \theta_2,...,\theta_n$. 

Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.

![simultaneously_update](../img/Simultaneous%20update.png)

# Gradient Descent Intuition
In this video we explored the scenario where we used one parameter $\theta_1$ and plotted its cost function to implement a gradient descent. 
Our formula for a single parameter was :

Repeat until convergence: 
$\quad \theta_{1}:=\theta_{1}-\alpha \frac{d}{d \theta_{1}} J\left(\theta_{1}\right)$

Regardless of the slope's sign for $\frac{d}{d\theta_1}$, $\theta_1$ eventually converges to its minimum value. 

The following graph shows that when the slope is negative, the value of $\theta_1$ increases and when it is positive, 
the value of $\theta_1$ decreases.

![img](../img/J(θ1).png)

On a side note, we should adjust our parameter $\alpha$ to ensure that the gradient descent algorithm converges in a reasonable time(在一个合理的时间内收敛). 

**Failure to converge** or **too much time** to obtain the minimum value imply that our **step size is wrong**.

![step size is wrong](../img/step%20size%20is%20wrong.png)


## How does gradient descent converge with a fixed step size $\alpha$?

The intuition behind the convergence is that $\frac{d}{d\theta_1}$ approaches 0 as we approach the bottom of our convex function. 

At the minimum, the derivative will always be 0 and thus we get:

$\theta_{1}:=\theta_{1}-\alpha * 0$

![gradient descent converge with a fixed step size α](../img/gradient%20descent%20converge%20with%20a%20fixed%20step%20size%20α.png)

# Gradient Descent in Practice

## Feature Scaling

We can **speed up gradient descent** by having each of our **input values in roughly the same range**. 

This is because θ will descend quickly on small ranges and slowly on large ranges, 
and so will oscillate(摆动) inefficiently down to the optimum(最佳) when the variables are very uneven(不均匀).
>因此，当变量非常不均匀时，将无法有效率的振荡到最佳状态。

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. 

Ideally:

$−1 ≤ x_{(i)}≤ 1$

or

$−0.5 ≤ x_{(i)}≤ 0.5$

These aren't exact requirements; 
we are only trying to **speed things up**. 

The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are [[feature-scaling]] and [[mean-normalization]]. 


## Learning Rate

**Debugging gradient descent**

Make a plot with number of iterations on the x-axis. 

Now plot the cost function, J(θ) over the number of iterations of gradient descent. 

If J(θ) ever **increases**, then you probably need to **decrease α**.


**Automatic convergence test**

Declare convergence if J(θ) decreases by less than E in one iteration, 
where E is some small value such as $10^{−3}$. 

However in practice it's difficult to choose this threshold value.

![learning rate](../img/Learning%20rate.png)

To summarize:

If $\alpha$ is too small: slow convergence.

If $\alpha$ is too large: ￼may not decrease on every iteration and thus may not converge.


---


- [[gradient-descent-for-linear-regression]]
- [[gradient-descent-for-logistic-regression.md]]


[//begin]: # "Autogenerated link references for markdown compatibility"
[feature-scaling]: feature-scaling "Feature Scaling"
[mean-normalization]: mean-normalization "Mean Normalization"
[gradient-descent-for-linear-regression]: gradient-descent-for-linear-regression "Gradient Descent For Linear Regression"
[//end]: # "Autogenerated link references"