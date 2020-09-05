# Neural Networks

## Model Representation
Let's examine how we will represent a hypothesis function using neural networks. 

At a very simple level, neurons are basically computational units that take inputs (dendrites '树突') as **electrical inputs** (called "spikes" '达到阈值时产生称为尖峰(spike)的短电脉冲') that are channeled to **outputs** (axons '轴突'). 

In our model, our dendrites are like the input features $x_1\cdots x_n$ , 
and the output is the result of our hypothesis function. 

In this model our $x_0$ input node is sometimes called the "**bias unit**." It is **always equal to 1**. 

In neural networks, we use the same **logistic function as in classification**, $\frac{1}{1 + e^{-\theta^Tx}}$, 
yet we sometimes call it a **sigmoid (logistic) activation function**. 

In this situation, our "theta" parameters are sometimes called "**weights**".

Visually, a simplistic representation looks like:

![a simplistic representation](../img/a%20simplistic%20representation%20of%20neural%20network.png)

Our input nodes (layer 1), also known as the "**input layer**", go into another node (layer 2), 
which finally outputs the hypothesis function, known as the "**output layer**".

We can have intermediate layers of nodes between the input and output layers called the "**hidden layers**."

In this example, we label these intermediate or "hidden" layer nodes $a^2_0 \cdots a^2_n$ and call them "**activation units**."

> $a_{i}^{(j)}="$ activation" of unit $i$ in layer $j$
>
> $\Theta^{(j)}=$ matrix of weights controlling function mapping from layer $j$ to layer $j+1$

If we had one hidden layer, it would look like:

![one hidden layer](../img/neural%20network%20with%20one%20hidden%20layer.png)

The values for each of the "activation" nodes is obtained as follows:

$$\begin{array}{r}
a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\
\\
a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\
\\
a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\
\\
h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right)
\end{array}$$

This is saying that we compute our activation nodes by **using a 3×4 matrix of parameters**. 

We apply each row of the parameters to our inputs to obtain the value for one activation node. 

Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, 
which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, $\Theta^{(j)}$.

The dimensions of these matrices of weights is determined as follows:

> If network has $s_{j}$ units in layer $j$ and $s_{j+1}$ units in layer $j+1,$ then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times\left(s_{j}+1\right)$.

The +1 comes from the addition in $\Theta^{(j)}$ of the "bias nodes," $x_0$ and $\Theta_0^{(j)}$. 

In other words **the output nodes will not include the bias nodes while the inputs will**. 

The following image summarizes our model representation:

![summarizes our model representation](../img/summarizes%20our%20model%20representation.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. 

Dimension of $\Theta^{(1)}$ is going to be 4×3 where $s_j = 2$ and $s_{j+1} = 4$, 
so $s_{j+1} \times (s_j + 1) = 4 \times 3$.

---
## Forward propagation
——前向传播

To re-iterate, the following is an example of a neural network:

$$\begin{array}{r}
a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\
\\
a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\
\\
a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\
\\
h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right)
\end{array}$$

In this section we'll do a vectorized implementation of the above functions. 

We're going to define a new variable $z_k^{(j)}$ that encompasses the parameters inside our g function. 

In our previous example if we replaced by the variable z for all the parameters we would get:

$a_{1}^{(2)}=g\left(z_{1}^{(2)}\right)$

$a_{2}^{(2)}=g\left(z_{2}^{(2)}\right)$

$a_{3}^{(2)}=g\left(z_{3}^{(2)}\right)$

In other words, for layer j=2 and node k, the variable z will be:

$z_{k}^{(2)}=\Theta_{k, 0}^{(1)} x_{0}+\Theta_{k, 1}^{(1)} x_{1}+\cdots+\Theta_{k, n}^{(1)} x_{n}$

The vector representation of x and $z^{j}$ is:

$x=\left[\begin{array}{l}x_{0} \\ x_{1} \\ \cdots \\ x_{n}\end{array}\right] z^{(j)}=\left[\begin{array}{c}z_{1}^{(j)} \\ z_{2}^{(j)} \\ \cdots \\ z_{n}^{(j)}\end{array}\right]$

Setting $x = a^{(1)}$, we can rewrite the equation as:

$z^{(j)}=\Theta^{(j-1)} a^{(j-1)}$

We are multiplying our matrix $\Theta^{(j-1)}$ with dimensions $s_j\times (n+1)$
(where $s_j$ is the number of our activation nodes) by our vector $a^{(j-1)}$ with height (n+1). 

This gives us our vector $z^{(j)}$ with height $s_j$. 

Now we can get a vector of our activation nodes for layer j as follows:

$a^{(j)}=g\left(z^{(j)}\right)$

Where our function g can be applied element-wise(按元素) to our vector $z^{(j)}$.

We can then add a bias unit (equal to 1) to layer j after we have computed $a^{(j)}$. 

This will be element $a_0^{(j)}$ and will be equal to 1. 

To compute our final hypothesis, let's first compute another z vector:

$z^{(j+1)}=\Theta^{(j)} a^{(j)}$

We get this final z vector by multiplying the next theta matrix after $\Theta^{(j-1)}$ with the values of all the activation nodes we just got. 

This last theta matrix $\Theta^{(j)}$ will have only one row which is multiplied by one column $a^{(j)}$ so that our result is a single number. 

We then get our final result with:

$h_{\Theta}(x)=a^{(j+1)}=g\left(z^{(j+1)}\right)$

Notice that in this **last step**, between layer j and layer j+1,
we are doing exactly the same thing as we did in [[logistic-regression]]. 

Adding all these intermediate layers(中间层) in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

![Forward propagation](../img/Forward%20propagation.png)

---

## Examples and Intuitions 
—— 实现逻辑与、或、非、同或功能

A simple example of applying neural networks is by predicting $x_1$ AND $x_2$, 
which is the logical **'and'** operator and is only true if both $x_1$ and $x_2$ are 1.

![AND](../img/AND.png)

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates. 

The following is an example of the logical operator **'OR'**, 
meaning either $x_1$ is true or $x_2$ is true, or both:

![OR](../img/OR.png)

![NOT](../img/NOT.png)

![XNOR](../img/XNOR.png)

---

- go to [[Cost Function for Neural Networks]]
- go to [[Classification]]



[//begin]: # "Autogenerated link references for markdown compatibility"
[logistic-regression]: logistic-regression "Logistic Regression"
[Classification]: classification "Classification"
[Cost Function for Neural Networks]:cost-function-for-neural-networks "Cost Function for Neural Networks"
[//end]: # "Autogenerated link references"