set terminal qt size 700,900

NNdir = "./Fri-08-22/NN/"
TNNdir = "./Thu-08-21/TNN/"
set datafile separator ","

set key font "CMU Serif, 20"
# set xrange [0:100]
set yrange [0:1]
set mxtics 2
set mytics 2
set tics out
set tics nomirror
# set yrange [0.4:0.9]
set lmargin 15
set rmargin 10
set bmargin 4
set xtics font "CMU Serif,20"
set ytics font "CMU Serif,20"
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "m_{eff}/m_{0}" offset -3,0 font "CMU Serif,30"
# plot dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb 'black' title "2q"


plot NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2002_MoS2_GGA_G.dat" u 2:5 w l lw 5 lc rgb "purple" notitle "",\

