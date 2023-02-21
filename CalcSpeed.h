#include <math.h>
inline void CalcSpeed(double *speed, const double *xd, const double *yd, const double *zd){
    speed[0] = sqrt((xd[0]*xd[0])+(yd[0]*yd[0])+(zd[0]*zd[0]));

}