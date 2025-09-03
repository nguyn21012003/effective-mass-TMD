set terminal qt size 700,900
# set terminal dumb 

NNdir = "./Sat-08-30/NN/"
TNNdir = "./Wed-09-03/TNN/"
set datafile separator ","

set key font "CMU Serif, 20"
# set xrange [0:100]
set yrange [0:1]
set ytics 0.05
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
set ylabel 'm_{eff}/m_{0}' offset -3,0 font "CMU Serif,30"
# plot dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb 'black' title "2q"

set grid

set title "GGA parameter" font "CMU Serif, 20"

plot  TNNdir . "Mass_q_297_MoS2_GGA.dat"  u 2:3 w l lw 3 lc rgb "light-red" title "m_{h} TNN case",\
      NNdir . "Mass_q_1341_MoS2_GGA.dat" u 2:3 w l lw 3 lc rgb "green" title "m_{h} NN case",\
      TNNdir . "Mass_q_297_MoS2_GGA.dat" u 2:4 w l lw 3 lc rgb "navy" title "m_{e} TNN case",\
      NNdir . "Mass_q_1341_MoS2_GGA.dat" u 2:4 w l lw 3 lc rgb "magenta" title "m_{e} NN case",\
      TNNdir . "Mass_q_297_MoS2_GGA.dat" u 2:3 with points pt 4 ps 2 lc rgb "light-red" notitle ,\
      TNNdir . "Mass_q_297_MoS2_GGA.dat" u 2:4 with points pt 5 ps 2 lc rgb "navy" notitle ,\
      NNdir . "Mass_q_1341_MoS2_GGA.dat" u 2:3 with points pt 8 ps 2 lc rgb "green" notitle "m_{e} NN case",\
      NNdir . "Mass_q_1341_MoS2_GGA.dat" u 2:4 with points pt 11 ps 2 lc rgb "magenta" notitle "m_{h} NN case",\

