# PythonInterpreter

## Examples

**Use the tool directly (without agent)**

```python
    from agentlego.apis import load_tool 
    tool = load_tool('PythonInterpreter')
    tool(
    """
    ```python
    import math
    def solution():
        x = 5
        y = math.log(x)
        return y
    ```
    """
    )
```

**With Lagent**

```python

```

# Solver

## Examples

**Use the tool directly (without agent)**

```python
    from agentlego.apis import load_tool
    tool = load_tool('Solver')
    tool(
    """
    ```python
    from sympy import symbols, Eq, solve
    def solution():
        x, y = symbols('x y')
        equation1 = Eq(x**2 + y**2, 20)
        equation2 = Eq(x**2 - 5*x*y + 6*y**2, 0)
        solutions = solve((equation1, equation2), (x, y))
        solutions_str = [f'x = {sol[0]}, y = {sol[1]}' for sol in solutions]
        return solutions_str
    ```
    """
    )
```

**With Lagent**

```python

```


# Plot

## Examples

**Use the tool directly (without agent)**

```python
    from agentlego.apis import load_tool
    tool = load_tool('Plot')
    tool(
    """
    ```python
    import matplotlib.pyplot as plt
    def solution(path):
        # labels and data
        cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
        data = [23, 17, 35, 29, 12, 41]
        # draw diagrams
        plt.figure(figsize=(8, 6))
        plt.pie(data, labels=cars, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Car Distribution')
        # save diagrams
        plt.savefig(path)
        return path
    ```
    """
    )
```

**With Lagent**

```python

```

