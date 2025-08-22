set terminal qt size 420,900

NNdir = "./Fri-08-22/NN/"
TNNdir = "./Thu-08-21/TNN/"
set datafile separator ","

set key font "CMU Serif, 20"
set xrange [0:250]

set lmargin 13
set rmargin 13
set bmargin 4
set xtics font "CMU Serif,20"
set ytics font "CMU Serif,20"
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "Energy (eV)" offset -2,0 font "CMU Serif,20"
# plot dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb 'black' title "2q"


plot NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:5 w points pt 7 ps 0.5 lc rgb "purple" title "omega_{c}",\

