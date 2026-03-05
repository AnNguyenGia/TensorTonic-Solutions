def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    tol = 1e-6
    for  _ in range(steps):
        update = 2*a*x0 + b
        if abs(update) < tol:
            break
        x0 -= update*lr

    return x0
        