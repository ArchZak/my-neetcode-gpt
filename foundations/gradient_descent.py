class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        #input: iterations, how many gradient iterations duh, learning rate so alpha, starting point init
        #output: whatever the output is but with round() for 5 dcecimal places

        #minimize function x^2 using gradient descent
        #need to arrive at x=0 iteratively
        #so 2x in this case, and then 2, and then 0

        #x new = x old - alpha * f'(xold) where alpha is learning rate
        #so do the update rule with the specific number of iterations on init

        #need a for loop for iterations
        
        for _ in range(iterations):
            init = init - learning_rate*2*init

        return round(init,5)
        
