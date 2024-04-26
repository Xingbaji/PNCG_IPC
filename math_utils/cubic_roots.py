# 这部分代码本来用于实现local CCD，但是好像有bug，local CCD的结果不对。
import taichi as ti

@ti.func
def cubic_roots(coef, x0, x1, tol=6e-4):
    """
    implemets cubic roots as https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h
    Finds the roots of the cubic polynomial between x0 and x1 with tol and returns the roots.
    :param coef: vector([d,c,b,a]) f = a * x*x*x + b * x*x + c * x + d
    :param x0: x_min
    :param x1: x_max
    :param tol:
    :return: vector([root0,root1,root2]) if there are less than 3 roots, return 10
    """
    ret = False
    # roots = ti.Vector([10,10,10], ti.f32)
    roots_0 = 10.0
    roots_1 = 10.0
    roots_2 = 10.0
    y0 = cubic_eval(coef, x0)
    y1 = cubic_eval(coef, x1)
    a = coef[3] *3
    b_2 = coef[2]
    c = coef[1]
    deriv = ti.Vector([c, 2*b_2, a, 0])
    delta_4 = b_2*b_2 - a*c
    # print('delta_4', delta_4)
    if delta_4 > 0:
        d_2 = ti.sqrt(delta_4)
        q = - ( b_2 + d_2 * NewSign(b_2) )
        rv0 = q / a
        rv1 = c / q
        xa = ti.min(rv0, rv1)
        xb = ti.max(rv0, rv1)
        # print('xa', xa, 'xb', xb, 'y0', y0, 'y1', y1, 'IsDifferentSign(y0,y1)', IsDifferentSign(y0,y1))
        if IsDifferentSign(y0,y1):
            if xa >= x1 or xb <= x0 or ( xa <= x0 and xb >= x1 ):
                roots_0 = FindClosed(coef, deriv, x0, x1, y0, tol)
                ret = True
        else:
            if (xa >= x1 or xb <= x0) or ( xa <= x0 and xb >= x1 ):
                ret = True

        if ret == False:
            if xa > x0:
                ya = cubic_eval(coef, xa)
                if IsDifferentSign(y0,ya):
                    roots_0 = FindClosed(coef, deriv, x0, xa, y0, tol)
                    if IsDifferentSign(ya,y1) or (xb < x1 and IsDifferentSign(ya, cubic_eval(coef, xb))):
                        defPoly0,defPoly1,defPoly2 = PolynomialDeflate(coef, roots_0)
                        roots_1, roots_2 = QuadraticRoots(defPoly0, defPoly1, defPoly2, xa, x1)
                elif xb < x1:
                    yb = cubic_eval(coef, xb)
                    if IsDifferentSign(ya,yb):
                        roots_0 = FindClosed(coef, deriv, xa, xb, ya, tol)
                        if IsDifferentSign(yb,y1):
                            defPoly0, defPoly1, defPoly2 = PolynomialDeflate(coef, roots_0)
                            roots_1, roots_2 = QuadraticRoots(defPoly0, defPoly1, defPoly2, xb, x1)
                    elif IsDifferentSign(yb,y1):
                        roots_0 = FindClosed(coef, deriv, xb, x1, yb, tol)
                elif IsDifferentSign(ya,y1):
                    roots_0 = FindClosed(coef, deriv, xa, x1, ya, tol)
            else:
                yb = cubic_eval(coef, xb)
                if IsDifferentSign(y0,yb):
                    roots_0 = FindClosed(coef, deriv, x0, xb,  y0, tol)
                    if IsDifferentSign(yb,y1):
                        defPoly0, defPoly1, defPoly2 = PolynomialDeflate(coef, roots_0)
                        roots_1, roots_2 = QuadraticRoots(defPoly0, defPoly1, defPoly2, xb, x1)
                elif IsDifferentSign(yb,y1):
                    roots_0 = FindClosed(coef, deriv, xb, x1, yb, tol)
    elif IsDifferentSign(y0,y1):
        roots_0 = FindClosed(coef, deriv, x0,  x1, y0, tol)
    return roots_0,roots_1,roots_2

@ti.func
def cubic_first_root(coef, x0, x1, tol=6e-4):
    """
    Finds the first root of the cubic polynomial between x0 and x1 with tol and returns the root.
    :param coef: vector([d,c,b,a]) f = a * x*x*x + b * x*x + c * x + d
    !!check if it returns the minimal root
    :param x0: x_min
    :param x1: x_max
    :param tol:
    :return: float
    """
    ret = False
    root = 10.0
    y0 = cubic_eval(coef, x0)
    y1 = cubic_eval(coef, x1)
    a = coef[3] * 3
    b_2 = coef[2]
    c = coef[1]
    deriv = ti.Vector([c, 2*b_2, a, 0])
    delta_4 = b_2*b_2 - a*c
    if delta_4 > 0:
        d_2 = ti.sqrt(delta_4)
        q = - ( b_2 + d_2 * NewSign(b_2) )
        rv0 = q / a
        rv1 = c / q
        xa = ti.min(rv0, rv1)
        xb = ti.max(rv0, rv1)
        if IsDifferentSign(y0,y1):
            if xa >= x1 or xb <= x0 or ( xa <= x0 and xb >= x1 ):
                root = FindClosed(coef, deriv, x0, x1, y0, tol)
                ret = True
        else:
            if (xa >= x1 or xb <= x0) or ( xa <= x0 and xb >= x1 ):
                ret = True

        if ret == False:
            if xa > x0:
                ya = cubic_eval(coef, xa)
                if IsDifferentSign(y0,ya):
                    root = FindClosed(coef, deriv, x0, xa, y0, tol)
                elif xb < x1:
                    yb = cubic_eval(coef, xb)
                    if IsDifferentSign(ya,yb):
                        root = FindClosed(coef, deriv, xa, xb, ya, tol)
                    elif IsDifferentSign(yb,y1):
                        root = FindClosed(coef, deriv, xb, x1, yb, tol)
                elif IsDifferentSign(ya,y1):
                    root = FindClosed(coef, deriv, xa, x1, ya, tol)
            else:
                yb = cubic_eval(coef, xb)
                if IsDifferentSign(y0,yb):
                    root = FindClosed(coef, deriv, x0, xb,  y0, tol)
                elif IsDifferentSign(yb,y1):
                    root = FindClosed(coef, deriv, xb, x1, yb, tol)
    elif IsDifferentSign(y0,y1):
        root = FindClosed(coef, deriv, x0,  x1, y0, tol)
    return root

@ti.func
def CubicHasRoot(coef, x0, x1):
    ret = 0
    y0 = cubic_eval(coef, x0)
    y1 = cubic_eval(coef, x1)
    if IsDifferentSign(y0,y1):
        ret = 1
    else:
        a = coef[3] *3
        b_2 = coef[2]
        c = coef[1]
        delta_4 = b_2*b_2 - a*c
        if delta_4 > 0:
            d_2 = ti.sqrt(delta_4)
            q = - (b_2 + d_2 * NewSign(b_2))
            rv0 = q / a
            rv1 = c / q
            xa = ti.min(rv0, rv1)
            xb = ti.max(rv0, rv1)
            if (xa >= x1 or xb <= x0) or ( xa <= x0 and xb >= x1 ):
                ret = 0
            elif xa > x0:
                ya = cubic_eval(coef, xa)
                if IsDifferentSign(y0,ya):
                    ret = 1
                elif xb < x1:
                    yb = cubic_eval(coef, xb)
                    if IsDifferentSign(y0,yb):
                        ret = 1
            elif xa <= x0:
                yb = cubic_eval(coef, xb)
                if IsDifferentSign(y0,yb):
                    ret = 1
    return ret



@ti.func
def QuadraticRoots(defPoly0, defPoly1, defPoly2, x0, x1):
    roots_0 = 10.0
    roots_1 = 10.0
    c = defPoly0
    b = defPoly1
    a = defPoly2
    delta = b*b - 4*a*c
    if delta > 0:
        d = ti.sqrt(delta)
        q = -0.5 * (b + d * NewSign(b))
        rv0 = q / a
        rv1 = c / q
        r0 = ti.min(rv0, rv1)
        r1 = ti.max(rv0, rv1)
        if (r0 >= x0) and (r0 <= x1):
            roots_0 = r0
        if (r1 >= x0) and (r1 <= x1):
            roots_1 = r1
    elif delta == 0:
        r0 = -0.5 * b / a
        if (r0 >= x0) and (r0 <= x1):
            roots_0 = r0
    return roots_0, roots_1

@ti.func
def PolynomialDeflate(coef, root):
    defPoly2 = coef[3]
    defPoly1 = coef[2] + root * defPoly2
    defPoly0 = coef[1] + root * defPoly1
    return defPoly0,defPoly1,defPoly2

@ti.func
def cubic_eval(coef, x):
    return x * (x * ( coef[3] * x + coef[2]) + coef[1]) + coef[0]

@ti.func
def NewSign(x):
    return ti.cast( (x >= 0) - (x<0) , ti.f32)
    # ( T v, S sign ) { return v * (sign<0 ? T(-1) : T(1)); }

@ti.func
def IsDifferentSign(a, b):
    return (a<0) != (b<0)

@ti.func
def FindClosed(coef, deriv, x0, x1, y0, xError):
    ep2 = 2 * xError
    xr = (x0 + x1) / 2
    ret = False
    if x1 - x0 > ep2:
        xr0 = xr
        for safetyCounter in range(16):
            xn = xr - cubic_eval(coef, xr) / cubic_eval(deriv, xr)
            xn = ti.max(x0, ti.min(x1, xn))
            if abs(xr - xn) <= xError:
                ret = True
                xr = xn
                break
                # return xn
            xr = xn
        if ret == False:
            if not ti.math.isinf(xr):
                xr = xr0

            yr = cubic_eval(coef, xr)
            xb0 = x0
            xb1 = x1
            while True:
                side = IsDifferentSign(y0,yr)
                if side:
                    xb1 = xr
                else:
                    xb0 = xr
                dy = cubic_eval(deriv, xr)
                dx = yr / dy
                xn = xr - dx
                if (xn > xb0) and (xn < xb1):
                    stepsize = ti.abs(xr - xn)
                    xr = xn
                    if stepsize > xError:
                        yr = cubic_eval(coef, xr)
                    else:
                        break
                else:
                    xr = (xb0 + xb1) / 2
                    if (xr == xb0) or (xr == xb1) or (xb1 - xb0 <= ep2):
                        break
                    yr = cubic_eval(coef, xr)
    return xr

@ti.kernel
def print_cubic_roots():
    x0 = 0.0
    x1 = 1.0
    coef = ti.Vector([-0.04, 0.53, -1.4, 1], ti.f32)
    # print(cubic_eval(coef, 0.1))
    # print(cubic_eval(coef, 0.5))
    # print(cubic_eval(coef, 0.8))
    # roots = cubic_roots(coef, x0, x1, tol=1e-3)
    # print(roots)
    root = cubic_first_root(coef, x0, x1, tol=1e-3)
    # roots = test0()
    # print(root)

@ti.func
def test0():
    roots = ti.Vector([1,2,3,4], ti.f32)
    roots = test1(roots)
    return roots

@ti.func
def test1(roots):
    roots[1] = -1
    if True:
        roots[3] = -2
    return roots


if __name__ == '__main__':
    ti.init(ti.gpu, default_fp=ti.f32, kernel_profiler=True)
    print_cubic_roots()
    ti.profiler.clear_kernel_profiler_info()
    for i in range(100000):
        print_cubic_roots()
    ti.profiler.print_kernel_profiler_info()


# template <int N, typename ftype, bool boundError>
# inline ftype RootFinderNewton::FindClosed( ftype const coef[N+1], ftype const deriv[N], ftype x0, ftype x1, ftype y0, ftype y1, ftype xError )
# {
# 	ftype ep2 = 2*xError;
# 	ftype xr = (x0 + x1) / 2;	// mid point
# 	if ( x1-x0 <= ep2 ) return xr;
#
# 	if constexpr ( N <= 3 ) {
# 		ftype xr0 = xr;
# 		for ( int safetyCounter=0; safetyCounter<16; ++safetyCounter ) {
# 			ftype xn = xr - PolynomialEval<N,ftype>( coef, xr ) / PolynomialEval<2,ftype>( deriv, xr );
# 			xn = Clamp( xn, x0, x1 );
# 			if ( std::abs(xr - xn) <= xError ) return xn;
# 			xr = xn;
# 		}
# 		if ( ! IsFinite(xr) ) xr = xr0;
# 	}
#
# 	ftype yr = PolynomialEval<N,ftype>( coef, xr );
# 	ftype xb0 = x0;
# 	ftype xb1 = x1;
#
# 	while ( true ) {
# 		int side = IsDifferentSign( y0, yr );
# 		if ( side ) xb1 = xr; else xb0 = xr;
# 		ftype dy = PolynomialEval<N-1,ftype>( deriv, xr );
# 		ftype dx = yr / dy;
# 		ftype xn = xr - dx;
# 		if ( xn > xb0 && xn < xb1 ) { // valid Newton step
# 			ftype stepsize = std::abs(xr-xn);
# 			xr = xn;
# 			if ( stepsize > xError ) {
# 				yr = PolynomialEval<N,ftype>( coef, xr );
# 			} else {
# 				if constexpr ( boundError ) {
# 					ftype xs;
# 					if ( xError == 0 ) {
# 						xs = std::nextafter( side?xb1:xb0, side?xb0:xb1 );
# 					} else {
# 						xs = xn - MultSign( xError, side-1 );
# 						if ( xs == xn ) xs = std::nextafter( side?xb1:xb0, side?xb0:xb1 );
# 					}
# 					ftype ys = PolynomialEval<N,ftype>( coef, xs );
# 					int s = IsDifferentSign( y0, ys );
# 					if ( side != s ) return xn;
# 					xr = xs;
# 					yr = ys;
# 				} else break;
# 			}
# 		} else { // Newton step failed
# 			xr = (xb0 + xb1) / 2;
# 			if ( xr == xb0 || xr == xb1 || xb1 - xb0 <= ep2 ) {
# 				if constexpr ( boundError ) {
# 					if ( xError == 0 ) {
# 						ftype xm = side ? xb0 : xb1;
# 						ftype ym = PolynomialEval<N,ftype>( coef, xm );
# 						if ( std::abs(ym) < std::abs(yr) ) xr = xm;
# 					}
# 				}
# 				break;
# 			}
# 			yr = PolynomialEval<N,ftype>( coef, xr );
# 		}
# 	}
# 	return xr;
# }
# template <typename ftype, bool boundError, typename RootFinder>
# inline int CubicRoots( ftype roots[3], ftype const coef[4], ftype x0, ftype x1, ftype xError )
# {
# 	const ftype y0 = PolynomialEval<3,ftype>( coef, x0 );
# 	const ftype y1 = PolynomialEval<3,ftype>( coef, x1 );
#
# 	const ftype a   = coef[3]*3;
# 	const ftype b_2 = coef[2];
# 	const ftype c   = coef[1];
#
# 	const ftype deriv[4] = { c, 2*b_2, a, 0 };
#
# 	const ftype delta_4 = b_2*b_2 - a*c;
#
# 	if ( delta_4 > 0 ) {
# 		const ftype d_2 = Sqrt( delta_4 );
# 		const ftype q = - ( b_2 + MultSign( d_2, b_2 ) );
# 		ftype rv0 = q / a;
# 		ftype rv1 = c / q;
# 		const ftype xa = Min( rv0, rv1 );
# 		const ftype xb = Max( rv0, rv1 );
#
# 		if ( IsDifferentSign(y0,y1) ) {
# 			if ( xa >= x1 || xb <= x0 || ( xa <= x0 && xb >= x1 ) ) {	// first, last, or middle interval only
# 				roots[0] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, x0, x1, y0, y1, xError );
# 				return 1;
# 			}
# 		} else {
# 			if ( ( xa >= x1 || xb <= x0 ) || ( xa <= x0 && xb >= x1 ) ) return 0;
# 		}
#
# 		int numRoots = 0;
# 		if ( xa > x0 ) {
# 			const ftype ya = PolynomialEval<3,ftype>( coef, xa );
# 			if ( IsDifferentSign(y0,ya) ) {
# 				roots[0] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, x0, xa, y0, ya, xError );	// first interval
# 				if constexpr ( !boundError ) {
# 					if ( IsDifferentSign(ya,y1) || ( xb < x1 &&  IsDifferentSign( ya, PolynomialEval<3,ftype>(coef,xb) ) ) ) {
# 						ftype defPoly[4];
# 						PolynomialDeflate<3>( defPoly, coef, roots[0] );
# 						return QuadraticRoots( roots+1, defPoly, xa, x1 ) + 1;
# 					} else return 1;
# 				} else numRoots++;
# 			}
# 			if ( xb < x1 ) {
# 				const ftype yb = PolynomialEval<3,ftype>( coef, xb );
# 				if ( IsDifferentSign(ya,yb) ) {
# 					roots[ !boundError ? 0 : numRoots++ ] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, xa, xb, ya, yb, xError );
# 					if constexpr ( !boundError ) {
# 						if ( IsDifferentSign(yb,y1) ) {
# 							ftype defPoly[4];
# 							PolynomialDeflate<3>( defPoly, coef, roots[0] );
# 							return QuadraticRoots( roots+1, defPoly, xb, x1 ) + 1;
# 						} else return 1;
# 					}
# 				}
# 				if ( IsDifferentSign(yb,y1) ) {
# 					roots[ !boundError ? 0 : numRoots++ ] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, xb, x1, yb, y1, xError );	// last interval
# 					if constexpr ( !boundError ) return 1;
# 				}
# 			} else {
# 				if ( IsDifferentSign(ya,y1) ) {
# 					roots[ !boundError ? 0 : numRoots++ ] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, xa, x1, ya, y1, xError );
# 					if ( !boundError ) return 1;
# 				}
# 			}
# 		} else {
# 			const ftype yb = PolynomialEval<3,ftype>( coef, xb );
# 			if ( IsDifferentSign(y0,yb) ) {
# 				roots[0] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, x0, xb, y0, yb, xError );
# 				if constexpr ( !boundError ) {
# 					if ( IsDifferentSign(yb,y1) ) {
# 						ftype defPoly[4];
# 						PolynomialDeflate<3>( defPoly, coef, roots[0] );
# 						return QuadraticRoots( roots+1, defPoly, xb, x1 ) + 1;
# 					} else return 1;
# 				}
# 				else numRoots++;
# 			}
# 			if ( IsDifferentSign(yb,y1) ) {
# 				roots[ !boundError ? 0 : numRoots++ ] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, xb, x1, yb, y1, xError );	// last interval
# 				if constexpr ( !boundError ) return 1;
# 			}
# 		}
# 		return numRoots;
#
# 	} else {
# 		if ( IsDifferentSign(y0,y1) ) {
# 			roots[0] = RootFinder::template FindClosed<3,ftype,boundError>( coef, deriv, x0, x1, y0, y1, xError );
# 			return 1;
# 		}
# 		return 0;
# 	}
# }