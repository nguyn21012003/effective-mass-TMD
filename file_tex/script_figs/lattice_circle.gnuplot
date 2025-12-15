set terminal qt size 900,900

set parametric
set size square
set xrange [-3:3]
set yrange [-3:3]
unset key
unset tics
unset border


N = 16
R1 = 1
R2 = 2
R3 = 3

array x1[N]
array y1[N]

array x2[N]
array y2[N]

array x3[N]
array y3[N]

# set samples 200

do for [i=1:N] {
    x1[i] = R1*cos(2*pi*i/N)
    y1[i] = R1*sin(2*pi*i/N)

    x2[i] = R2*cos(2*pi*i/N)
    y2[i] = R2*sin(2*pi*i/N)

    x3[i] = R3*cos(2*pi*i/N)
    y3[i] = R3*sin(2*pi*i/N)

}

set label "n=1" at -0.25,0.75 font "CMU Serif,28"
set label "n=2" at -0.25,1.75 font "CMU Serif,28"
set label "n=3" at -0.25,2.75 font "CMU Serif,28"
set label "(b)" at -2.5,2.75 font "CMU Serif,28"

plot for [n=1:3] n*cos(t), n*sin(t) w l lw 2 lc rgb "black" notitle, \
     for [i=1:N] x1[i], y1[i] with points pt 7 ps 2 lc rgb "#1a80bb" notitle,\
     for [i=1:N] x2[i], y2[i] with points pt 7 ps 2 lc rgb "#1a80bb" notitle,\
     for [i=1:N] x3[i], y3[i] with points pt 7 ps 2 lc rgb "#1a80bb" notitle
