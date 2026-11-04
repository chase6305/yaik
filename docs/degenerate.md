# Degenerate Case in IK Solving

## Unary Linear Equation

Consider a very simple linear equation *ax = b*, where *x* is the unknown, *a, b* are the linear coefficient which might
depend on the ik input (end-effector pose) or solved unknowns. In other words, *a, b* are not necessarily constants.

From the equation above, we can get a solution to unknown *x = b/a* if *a* is not zero. When *a = 0*, we can conclude
no solution if *b* is not zero. If *a = 0* **AND** *b = 0*, the equation *ax = b* provides us no information 
about *x*, because it is *0 = 0*. In this case, we must **resort to other equations** to 1) find the solution of *x*; 
or 2) conclude there is no solution. We claim this situation that we must resort to other equations as the
*degeneration* of the original equation *ax = b*.

From the analysis above, the unary linear equation *ax = b* degenerate if and only if *a = 0* **AND** *b = 0*. This is
different from the case that the solution *x = b/a* becomes undefined, which only requires *a = 0*. This can eliminate
a lot of branches in the solution tree (because we can safely conclude no solution).

## Unary Sin/Cos Equation

Consider the equation in the form of *a sin(theta) + b cos(theta) + c = 0*, where *theta* is the unknown. The 
coefficient *a, b, c* defines a line in the xy plane, and *theta* is the intersection of that line with the unit
circle. As long as *a, b, c* defines a line, there might be zero/one/two intersection points, which implies *theta*
has no/one/two solution(s).

The degenerate case is that *a, b, c* **DO NOT DEFINE A LINE**. If *a=b=c=0*, then *a sin(theta) + b cos(theta) + c = 0*
is the entire 2d plane. In this situation, we must resort to other equations to 1) find the solution of *theta*; 
or 2) conclude there is no solution.

Thus, the equation *a sin(theta) + b cos(theta) + c = 0* degenerate if and only if *a = 0* **AND** *b = 0* **AND** *c=0*.
Note that this covers *a sin(theta) + c = 0* as a special case.


## Binary Sin/Cos Equation by Tangent Solver

We're interested in the equations:

Equation 1: *A sin(theta) + B = 0*   
Equation 2: *C cos(theta) + D = 0*

In this case, we would like to obtain a solution in the form of atan2(sin_term = - B / A, cos_term = - C / D). On the
contrary, if we only use Equation 1 or 2, we will have two solutions (and one of them is invalid).

If any of the equation degenerate (thus, provides no information), then we should not solve **IN THIS FORM**. For 
instance, if *A=0* **AND** *B=0*, then we should directly solve Equation 2 (and may get two solutions). Thus, these
equations 1 && 2 degenerate if *(A=0* **AND** *B=0)* **OR** *(C=0* **AND** *D=0)*.

In our implementation, we resort to other equation if *A=0* **OR** *C=0*, which can be written as *A * C = 0*. 
In other words, we use a more conservative approach and resort to other equations more than necessary.

## Linear Solver Type 1

We're interested in the equations:

Equation 1: *A sin(theta) + B cos(theta) = C*   
Equation 2: *-B sin(theta) + A cos(theta) = D*

We can think of *sin(theta), cos(theta)* as the solution of a linear system. This is a very special linear system such
that its linear coefficient is [A B; -B A]. The system doesn't have unique solution if and only if *A^2 + B^2 = 0*, 
which implies *A = 0* **AND** *B = 0*.

## Linear Solver Type 2

Each row of this system is *a sin(x) + b cos(x) + c sin(y) + d cos(y) = 0*.
Two independent rows express *[sin(x), cos(x)] = B [sin(y), cos(y)]*. A third row
then constrains *y*. If the selected rows become dependent, the numerical helper
tries another independent pair and the remaining row with the largest elimination
residual, using all available equations before declaring a continuous family.

When the eliminated equations vanish and *B* is orthogonal, every value of *y*
preserves the unit-circle constraint. The helper returns *0* and *pi* as family
representatives. These two values do not enumerate the family or guarantee a
solution inside application-specific joint limits. Other rank-deficient cases
remain unsupported, and every returned IK candidate still requires FK validation.

For real-robot validation, compare candidates against independent model FK after
converting the base and tool frames. Check position error, rotation error, joint
limits, and preservation of any fixed redundant joint separately. A round trip
through generated FK alone cannot detect a shared model or frame error.

Code generation reports the number of unimplemented degenerate branches and exposes
their node indices and conditions through `unhandled_degenerate_branches`. This is a
structural report, not a count of reachable failures. Candidates dropped at such a
branch are excluded from raw output if their actual robot joints are still incomplete;
already complete joint values are retained even when an auxiliary variable is unresolved.
