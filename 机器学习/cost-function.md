# Cost Function
We can measure the accuracy of our hypothesis function by using a **cost function**. 

This takes an average difference (actually a fancier version of an average)
of all the results of the hypothesis with inputs from x's and the actual output y's.

$J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}$

To break it apart, it is $\dfrac{1}{2} \bar x$
where $\bar{x}$  is the mean of the squares of $h_{\theta}\left(x_{i}\right)-y_{i}$, 
or the **difference** between the **predicted value** and the **actual value**.

This function is otherwise called the **"Squared error function"**(平方误差函数), or "Mean squared error". 
The mean is halved ${\dfrac{1}{2}}$ as a convenience for the computation of the gradient descent, 
as the derivative term of the square function will cancel out the $\dfrac{1}{2}$ term. 
The following image summarizes what the cost function does:
![Cost Function](../img/Cost%20Function.png)
除以m是使得误差平均到每个样本，除以2是一个微积分技巧，用于消除计算偏导数时出现的2