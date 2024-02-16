import numpy as np

def gauss_seidel(A, b, w, tol=1e-6, max_iter=100):
  """
  Solves a system of linear equations Ax = b using the Gauss-Seidel method with relaxation.

  Args:
    A: A square numpy array representing the coefficient matrix.
    b: A numpy array representing the right-hand side vector.
    w: The relaxation parameter.
    tol: The tolerance for convergence.
    max_iter: The maximum number of iterations.

  Returns:
    x: A numpy array representing the approximate solution.
    k: The number of iterations required to converge.
  """
  n = len(A)
  x = np.zeros(n)
  k = 0

  while True:
    x_new = np.zeros(n)
    for i in range(n):
      sum = b[i]
      for j in range(i):
        sum -= A[i, j] * x_new[j]
      for j in range(i + 1, n):
        sum -= A[i, j] * x[j]
      x_new[i] = (1 - w) * x[i] + w * sum / A[i, i]
    
    # Check for convergence
    r = np.linalg.norm(A @ x_new - b)
    if r < tol:
      break
    
    k += 1
    if k >= max_iter:
      print("Warning: Maximum number of iterations reached.")
      break
    
    x = x_new

  return x, k

# Example usage
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])
b = np.array([0, 0, 0])
w = 1.0

x, k = gauss_seidel(A, b, w)

print("Solution:", x)
print("Number of iterations:", k)

#import numpy as np

#def gauss_seidel(A, b, x0, omega, tolerance=1e-6, max_iterations=1000):
 #   n = len(b)
  #  x = x0.copy()
   # k = 0
    #while k < max_iterations:
     #   x_prev = x.copy()
      #  for i in range(1, n-1):
       #     x[i] = (1 - omega) * x[i] + omega * (b[i] - A[i, :i].dot(x[:i]) - A[i, i+1:].dot(x_prev[i+1:])) / A[i, i]
        #residual = np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)
        #if residual < tolerance:
         #   break
        #k += 1
    #return x, k

#def exact_solution(n):
 #   return [-n/4 + i/2 for i in range(1, n+1)]

#def main():
 #   n = 20
  #  A = np.diag(np.full(n, 2)) + np.diag(np.full(n-1, -1), k=1) + np.diag(np.full(n-1, -1), k=-1)
   # b = np.zeros(n)
    #x0 = np.zeros(n)

    #exact_sol = exact_solution(n)
    #print("Exact solution:", exact_sol)

    # (i) Solve with omega = 1
    #omega = 1
    #x, iterations = gauss_seidel(A, b, x0, omega)
    #print(f"Solution with omega = {omega}: {x}")
    #print("Iterations:", iterations)

    # (ii) Comment on the convergence if each diagonal element is changed from 2 to 4 and 6
    #for diag_val in [4, 6]:
     #   A_changed = A.copy()
      #  np.fill_diagonal(A_changed, diag_val)
       # x, iterations = gauss_seidel(A_changed, b, x0, omega)
        #print(f"\nSolution with diagonal elements {diag_val}: {x}")
        #print("Iterations:", iterations)

    # (iii) Use different relaxation factors and comment on the convergence
    #for omega in [0.5, 0.8, 1.0, 1.2, 1.5]:
      #  x, iterations = gauss_seidel(A, b, x0, omega)
       # print(f"\nSolution with omega = {omega}: {x}")
        #print("Iterations:", iterations)

#if __name__ == "__main__":
    #main() 
