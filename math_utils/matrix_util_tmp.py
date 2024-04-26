import taichi as ti

@ti.func
def spd_matrix(D):
    A = ti.cast(D,float)
    N = A.n
    maxRot = 5*(N**2)
    p = ti.Matrix.identity(float,A.n)
    Iter = 0
    while Iter < maxRot:
    # for i in range(maxRot):
        Iter += 1
        k=0;l=1
        max_value = 0.0
        for i in range(N - 1):
            for j in range(i+1,N):
                if ti.abs(A[i,j])>max_value:
                    max_value = ti.abs(A[i,j])
                    k=i;l=j
        # print('max_value',max_value)
        if max_value < 1.0e-7:
            # print('iter', Iter)
            break
        ADiff = A[l,l] - A[k,k]
        temp = A[k, l]
        t = 0.0
        # if ti.abs(A[k,l]) < ti.abs(ADiff)*1.0e-36:
        #     t = A[k,l] / ADiff
        # else:
        if ti.abs(temp) >= ti.abs(ADiff)*1.0e-36:
            phi = ADiff / (2.0 * temp)
            t = 1.0 / (ti.abs(phi) + ti.sqrt(1.0 + phi**2))
            if phi < 0.0:
                t = -t
        c = 1.0 / ti.sqrt(t ** 2 + 1.0);
        s = t * c
        tau = s / (1.0 + c)
        A[k, l] = 0.0
        A[k, k] = A[k, k] - t * temp
        A[l, l] = A[l, l] + t * temp
        for i in range(k):  # Case of i < k
            temp = A[i, k]
            A[i, k] = temp - s * (A[i, l] + tau * temp)
            A[i, l] = A[i, l] + s * (temp - tau * A[i, l])
        for i in range(k + 1, l):  # Case of k < i < l
            temp = A[k, i]
            A[k, i] = temp - s * (A[i, l] + tau * A[k, i])
            A[i, l] = A[i, l] + s * (temp - tau * A[i, l])
        for i in range(l + 1, N):  # Case of i > l
            temp = A[k, i]
            A[k, i] = temp - s * (A[l, i] + tau * temp)
            A[l, i] = A[l, i] + s * (temp - tau * A[l, i])
        for i in range(N):  # Update transformation matrix
            temp = p[i, k]
            p[i, k] = temp - s * (p[i, l] + tau * p[i, k])
            p[i, l] = p[i, l] + s * (temp - tau * p[i, l])

    eig_value_matrix = ti.Matrix.zero(float,A.n,A.n)
    spd_matrix = ti.Matrix.zero(float,A.n,A.n)
    flag = 0
    for i in range(N):
        if A[i,i] < -1e-7:
            flag = 1
            eig_value_matrix[i, i] = 0.0
        else:
            eig_value_matrix[i, i] = A[i,i]
    if flag == 0:
        spd_matrix = D
    else:
        spd_matrix = p @ eig_value_matrix @ p.transpose()
    return spd_matrix


@ti.func
def flatten3x3(A):
    """flatten 3*3 to 9*1"""
    flattened = ti.Vector([A[0,0], A[1,0], A[2,0], A[0,1],A[1,1],A[2,1],A[0,2],A[1,2],A[2,2]],float)
    return flattened

@ti.func
def flatten_and_outproduct(A):
    #3x3
    a0 = A[0,0]
    a1 = A[1,0]
    a2 = A[2,0]
    a3 = A[0,1]
    a4 = A[1,1]
    a5 = A[2,1]
    a6 = A[0,2]
    a7 = A[1,2]
    a8 = A[2,2]
    B = ti.Matrix([[a0 * a0, a0 * a1, a0 * a2, a0 * a3, a0 * a4, a0 * a5, a0 * a6, a0 * a7, a0 * a8],
                   [a1 * a0, a1 * a1, a1 * a2, a1 * a3, a1 * a4, a1 * a5, a1 * a6, a1 * a7, a1 * a8],
                   [a2 * a0, a2 * a1, a2 * a2, a2 * a3, a2 * a4, a2 * a5, a2 * a6, a2 * a7, a2 * a8],
                   [a3 * a0, a3 * a1, a3 * a2, a3 * a3, a3 * a4, a3 * a5, a3 * a6, a3 * a7, a3 * a8],
                   [a4 * a0, a4 * a1, a4 * a2, a4 * a3, a4 * a4, a4 * a5, a4 * a6, a4 * a7, a4 * a8],
                   [a5 * a0, a5 * a1, a5 * a2, a5 * a3, a5 * a4, a5 * a5, a5 * a6, a5 * a7, a5 * a8],
                   [a6 * a0, a6 * a1, a6 * a2, a6 * a3, a6 * a4, a6 * a5, a6 * a6, a6 * a7, a6 * a8],
                   [a7 * a0, a7 * a1, a7 * a2, a7 * a3, a7 * a4, a7 * a5, a7 * a6, a7 * a7, a7 * a8],
                   [a8 * a0, a8 * a1, a8 * a2, a8 * a3, a8 * a4, a8 * a5, a8 * a6, a8 * a7, a8 * a8]
                   ],float)
    return B


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.func
def svd_rv(F):
    #文档中的实现方法，结果相同，但速度慢
    U, sig, V = ti.svd(F)
    L22 = (U @ (V.transpose())).determinant()
    L = ti.Matrix([[1,0,0],[0,1,0],[0,0,L22]],dt=float)
    detU = U.determinant()
    detV = V.determinant()
    if detU < 0.0 and detV > 0.0:
        U = U @ L
    if detU > 0.0 and detV < 0.0:
        V = V @ L
    sig[2,2] = sig[2,2] * L22
    return U, sig, V

@ti.kernel
def multiply(ans: ti.template(), k: float, b: ti.template()):
    for i in ans:
        ans[i] = k * b[i]

@ti.kernel
def divide(ans: ti.template(), a: ti.template(), b: ti.template()):
    for i in ans:
        ans[i] = a[i] / b[i]


@ti.kernel
def add(ans: ti.template(), a: ti.template(), k: float, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]

@ti.kernel
def add_test(ans: ti.template(), a: ti.template(),  b: ti.template()):
    for i in ans:
        ans[i] = a[i] + b[i]


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> float:
    ans = 0.0
    ti.loop_config(block_dim_adaptive=True)
    for i in a: ans += a[i].dot(b[i])
    return ans