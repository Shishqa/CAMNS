from cvxopt import matrix, solvers
import numpy as np
import time

solvers.options['show_progress'] = False


def _check_extreme(C, d, alpha, tol=1e-6):
    """
    Verify if alpha is an extreme point of the polyhedral set
        { a | Ca+ d => 0 }.

    Inputs:
        (C,d):    the set of polyhedron parameters;
        alpha:    is the point to be tested;
          tol:    specifies the numerical tolerance.
    
    Returns:
        1 if alpha is an extreme point, 0 otherwise.
    """
    L, D = C.shape
    
    T = C[np.all(np.abs(C.dot(alpha) + d) < tol, axis=1), :]
    if T.shape[0] == 0:
        return False

    return np.linalg.matrix_rank(T, tol=tol) == D


def CAMNS(X, N, tol_lp=1e-3, tol_ext=1e-6):    
    """
    A practical implementation of the CAMNS-LP method.
    
    Inputs:
             X:    the M-by-L observation matrix, where M is the number
                   of observations;
             N:    the number of sources;
        tol_lp:    the numerical tolerance for checking if LP problems
                   function values are close to zero;
       tol_ext:    the numerical tolerance for extreme point checking.
       
    Output:
             S:    the N-by-L extracted source matrix
    """
    total_start = time.time()
    
    M, L = X.shape

    X = X.T
    
    d = np.mean(X, axis=1, keepdims=True)

    C, _, _ = np.linalg.svd(X - d, full_matrices=False)
    C = C[:, :N-1]
    
    l = 0
    B = np.diag(np.ones(L))
    S = np.zeros((0, L))

    epoch = 1
    while l < N:
        print('epoch #{}:'.format(epoch))
        time_start = time.time()
        
        w = np.random.normal(loc=0, scale=1, size=L)
        r = B.dot(w)

        G = matrix(-C)
        h = matrix(d)
        c = matrix(-C.T.dot(r))

        sol = solvers.conelp(c, G, h)
        alpha_1 = np.array(sol['x'])
        vec_1 = C.dot(alpha_1) + d
        pstar = np.abs(r.T.dot(vec_1))

        sol = solvers.conelp(-c, G, h)
        alpha_2 = np.array(sol['x'])
        vec_2 = C.dot(alpha_2) + d
        qstar = np.abs(r.T.dot(vec_2))

        if l == 0:
            if _check_extreme(C, d, alpha_1, tol=tol_ext):
                S = np.append(S, [vec_1.squeeze()], axis=0)

            if _check_extreme(C, d, alpha_2, tol=tol_ext):
                S = np.append(S, [vec_2.squeeze()], axis=0)

        else:
            if pstar / np.linalg.norm(r) / np.linalg.norm(vec_1) >= tol_lp:
                if _check_extreme(C, d, alpha_1, tol=tol_ext):
                    S = np.append(S, [vec_1.squeeze()], axis=0)

            if qstar / np.linalg.norm(r) / np.linalg.norm(vec_2) >= tol_lp:
                if _check_extreme(C, d, alpha_2, tol=tol_ext):
                    S = np.append(S, [vec_2.squeeze()], axis=0)

        print('\t{} new vectors'.format(S.shape[0] - l))
        l = S.shape[0]
        print('\ttotal number of vectors: {}'.format(l))

        Q, R = np.linalg.qr(S.T)
        B = np.diag(np.ones(L)) - Q.dot(Q.T)
        
        print('\t{}s elapsed'.format(time.time() - time_start))
        epoch += 1
        
    print('finished after {}s'.format(time.time() - total_start))
    return S