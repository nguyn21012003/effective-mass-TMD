set terminal qt size 900,900

set parametric
set xrange [-6:6]
set yrange [-6:6]
unset key
unset tics
unset border

N_per_square = 40
N_R = 6
N_total = N_per_square * N_R

array x[N_total]
array y[N_total]
array color[N_total]

idx = 1

set arrow from 0,-5 to 0,5.5 nohead lc rgb "black" lw 2
set arrow from -5,0 to 5.5,0 nohead lc rgb "black" lw 2

set label "k_{x}" at 0.4,4.5 tc rgb "black" font "CMU Serif,28"
set label "k_{y}" at 5,-0.3 tc rgb "black" font "CMU Serif,28"
set label "(a)" at -2.7,4.5 font "CMU Serif,28"

do for [R=0:5] {
    do for [i=1:N_per_square] {
        j = i % (N_per_square/4)
        k = floor(i/(N_per_square/4))
        if (k==0) {
            x[idx] = -R + 2*R*j/(N_per_square/4)
            y[idx] = -R
        } else if (k==1) {
            x[idx] = R
            y[idx] = -R + 2*R*j/(N_per_square/4)
        } else if (k==2) {
            x[idx] = R - 2*R*j/(N_per_square/4)
            y[idx] = R
        } else {
            x[idx] = -R
            y[idx] = R - 2*R*j/(N_per_square/4)
        }
        color[idx] = R
        idx = idx + 1
    }
}

plot for [i=1:N_total] x[i], y[i] with points pt 7 ps 2 lc rgb "#1a80bb" notitle
