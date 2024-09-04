# Giunta-Function-Algorithms

La **Giunta Function** es una función de prueba utilizada en la optimización global, diseñada para evaluar el rendimiento de algoritmos de optimización. Es una función multimodal, lo que significa que tiene múltiples mínimos locales, lo que la hace un desafío interesante para los métodos de optimización. Es continua, diferenciable, separable y escalable.

## Definición

La Giunta Function se define matemáticamente como:

$$
f(x)=0.6+\sum_{i=1}^2​[\sin(\frac{16}{15}x_i-1​)+\sin^2(\frac{16}{15}x_i-1​) + \frac{1}{50}\sin(4(\frac{16}{15}x_i-1​))]
$$

sujeto a $-1 \le x_i \le 1$. El mínimo global se alzanza en el punto $x^* = f(0.45834282, 0.45834282)$ y el mínimo alcanzado es $f(x^*)=0.060447$.

## Algoritmos probados

- Minimize
- Differential Evolution
- Genetic Algorithm
- Trust Region
- Quasi-Newton
- Max Descent
- PSO
- Simulated Annealing
- Basin Hopping