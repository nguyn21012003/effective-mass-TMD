set terminal qt size 700,900 enhanced


NNdir = "./Fri-10-03/NN/"
TNNdir = "./Wed-09-03/TNN/"
set datafile separator ","
set xrange [10000:20000]

set key font "CMU Serif, 20"


set tics nomirror out
set xtics 5000 font "CMU Serif,20"
set ytics font "CMU Serif,20"
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "Energy (eV)" offset -2,0 font "CMU Serif,20"

plot NNdir . "Butterfly_q_297_MoS2_GGA.dat" u 2:3 w p pt 7 ps 0.05 notitle


