Download Link: https://programming.engineering/product/csc311-homework-3/

# CSC311-Homework-3
CSC311 Homework 3


Submission: You will need to submit three les:

Your answers to all of the questions, as a PDF le titled hw3_writeup.pdf. You can produce the le however you like (e.g. LATEX, Microsoft Word, scanner), as long as it is readable. If you need to split your writeup into multiple les, that’s OK, as long as we can gure out what you did.

The completed Python les naive_bayes.py and q4.py.

Neatness Point: One point will be given for neatness. You will receive this point as long as we don’t have a hard time reading your solutions or understanding the structure of your code.

Late Submission: 10% of the marks will be deducted for each day late, up to a maximum of 3 days. After that, no submissions will be accepted.

Homeworks are individual work. See the Course Information handout1 for detailed policies.

[5pts] Backprop

In this question, you will derive the backprop updates for a particular neural net architecture. The network is similar to the multilayer perceptron architecture from lecture, and has one linear hidden layer. However, there are two architectural di erences:

In addition to the usual vector-valued input x, there is a vector-valued \context” input . (The particular meaning of isn’t important for your derivation, but think of it as containing additional task information, such as whether to focus on the left or the right half of the image.) The hidden layer activations are modulated based on ; this means they are multiplied by a value which depends on .

The network has a skip connection which sends information directly from the input to the output of the network.

The loss function is squared error. The forward pass equations and network architecture are as follows (the symbol represents elementwise multiplication, and denotes the logistic function):


CSC311 Homework 3

[1pt] Draw the computation graph relating x, z, , s, h, and the model parameters.

[4pts] Derive the backprop formulas to compute the error signals for all of the model parameters, as well as x and . Also include the backprop formulas for all intermediate quantities needed as part of the computation. You may leave the derivative of the logistic function as 0 rather than expanding it out explicitly.

[13pts] Fitting a Na ve Bayes Model

In this question, we’ll t a Na ve Bayes model to the MNIST digits using maximum likeli-hood. In addition to the mathematical derivations, you will complete the implementation in naive_bayes.py. The starter code will download the dataset and parse it for you: Each training sample (t(i); x(i)) is composed of a vectorized binary image x(i) 2 f0; 1g784, and 1-of-10 encoded class label t(i), i.e., t(ci) = 1 means image i belongs to class c.

Given parameters and , Na ve Bayes de nes the joint probability of the each data point x and its class label c as follows:

784

Y

p(x; c j ; ) = p(c j ; )p(x j c; ; ) = p(c j ) p(xj j c; jc):

j=1

where p(c j ) = c and p(xj = 1 j c; ; ) = jc. Here, is a matrix of probabilities for each pixel and each class, so its dimensions are 784 10, and is a vector with one entry for each class. (Note that in the lecture, we simpli ed notation and didn’t write the probabilities conditioned on the parameters, i.e. p(cj ) is written as p(c) in lecture slides).

For binary data (xj 2 f0; 1g), we can write the Bernoulli likelihood as

p(xj j c; jc) = jcxj (1 jc)(1

xj );

(1)

which is just a way of expressing p(xj = 1jc; jc) = jc

and p(xj = 0jc; jc) = 1

jc in

a compact form. For the prior p(t j ), we use a categorical distribution (generalization of Bernoulli distribution to multi-class case),

9

9tj

Xi

p(tc = 1 j ) = p(c j ) = c or equivalently p(t j ) = j=0 j where

i = 1;

=0

where p(c j ) and p(t j ) can be used interchangeably. You will t the parameters and using MLE and MAP techniques. In both cases, your tting procedure can be written as a few simple matrix multiplication operations.

[3pts] First, derive the maximum likelihood estimator (MLE) for the class-conditional pixel probabilities and the prior . Derivations should be rigorous.

Hint 1: We saw in lecture that MLE can be thought of as ‘ratio of counts’ for the data,

^

so what should jc be counting?

Hint 2: Similar to the binary case, when calculating the MLE for j for j = 0; 1; :::; 8,

t(i)

write p(t(i) j ) = 9j=0 jj and in the log-likelihood replace 9 = 1 8j=0 j, and then take derivatives w.r.t. j. This will give you the ratio ^j=^9 for j = 0; 1; ::; 8. You know

that ^j’s sum up to 1.

(b) [1pt] Derive the log-likelihood log p(tjx; ; ) for a single training image.

CSC311

Homework 3

(c)

[3pt] Fit the parameters and using the training set with MLE, and try to report

1

N

(i)

jx

(i)

^

the average log-likelihood per data point

i=1 log p(t

; ; ^), using Equation (1).

N

What goes wrong? (it’s okay if you can’t compute the average log-likelihood here).

(d)

^

[1pt] Plot the MLE estimator as 10 separate greyscale images, one for each class.

(e)

[2pt] Derive the Maximum A posteriori Probability (MAP) estimator for the class-

conditional pixel probabilities , using a Beta(3, 3) prior on each jc. Hint: it has

a simple nal form, and you can ignore the Beta normalizing constant.

(f)

[2pt] Fit the parameters and using the training set with MAP estimators from previ-

1

N

(i)

jx

(i)

^

ous part, and report both the average log-likelihood per data point,

i=1 log p(t

; ;^),

N

and the accuracy on both the training and test set. The accuracy is de ned as the frac-

tion of examples where the true class is correctly predicted using c^ = argmaxc log p(tc =

^

1jx; ; ^).

(g)

^

[1pt] Plot the MAP estimator as 10 separate greyscale images, one for each class.

3. [7pts] Categorial Distribution. In this problem you will consider a Bayesian approach to modelling categorical outcomes. Let’s consider tting the categorical distribution, which is a discrete distribution over K outcomes, which we’ll number 1 through K. The probability of each category is explicitly represented with parameter k. For it to be a valid probability

P

distribution, we clearly need k 0 and k k = 1. We’ll represent each observation x as a 1-of-K encoding, i.e, a vector where one of the entries is 1 and the rest are 0. Under this model, the probability of an observation can be written in the following form:

K

p(xj ) = Y kxk :

k=1

Suppose you observe a dataset,

= fx(i)gNi=1:

Denote the count for outcome k as Nk =

n

(i)

i=1 xk

. Recall that each data point is in the

(i)

datapoint represents an outcome k and x(i)

= 0

1-of-K encoding, i.e., xk

= 1 if the ith

P

k

otherwise. In the previous assignment, you showed that the maximum likelihood estimate for the counts was:

Nk k = N :

[2pts] For the prior, we’ll use the Dirichlet distribution, which is de ned over the set of probability vectors (i.e. vectors that are nonnegative and whose entries sum to 1). Its PDF is as follows:

p( ) / 1a1 1 Kak 1:

Determine the posterior distribution p( j D). Based on your answer, is the Dirichlet distribution a conjugate prior for the categorial distribution?

[3pts] Still assuming the Dirichlet prior distribution, determine the MAP estimate of the parameter vector . For this question, you may assume each ak > 1.

P

Hint: Remember that you need to enforce the constraint that k k = 1. You can do this using the same parameterization trick you used in Question 2. Alternatively, you could use Lagrange multipliers, if you’re familiar with those.

CSC311 Homework 3

[2pts] Now, suppose that your friend said that they had a hidden N + 1st outcome, x(N+1), drawn from the same distribution as the previous N outcomes. Your friend does not want to reveal the value of x(N+1) to you. So, you want to use your Bayesian model to predict what you think x(N+1) is likely to be. The \proper” Bayesian predictor is the

so-called posterior predictive distribution:

Z

p(x(N+1)jD) = p(x(N+1)j )p( jD) d

