set terminal qt size 420,900

NNdir = "./Mon-08-25/NN/"
TNNdir = "./Sat-08-23/TNN/"
set datafile separator ","
# set xrange [0:50]
# set yrange [*:-0.05]

set key font "CMU Serif, 20"


set lmargin 13
set rmargin 13
set bmargin 4
set tics nomirror out
# set xtics 25 font "CMU Serif,20"
# set ytics 0.01 font "CMU Serif,20"
# set mytics 2
# set mxtics 5
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "Energy (eV)" offset -2,0 font "CMU Serif,20"
# plot dir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb 'black' notitle "2q"

# plot NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_4001_MoS2_GGA_G.dat" u 2:4 w lines lw 5 notitle,\
    NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2863_MoS2_GGA_G.dat" u 2:6 w lines lw 5 notitle,\
    NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_2863_MoS2_GGA_G.dat" u 2:8 w lines lw 5 notitle

plot for [i=4:80] NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:i with lines lw 1 lc rgb "#bdbdbd" notitle "2q+1",\
  # NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:4 w l lw 3 lc rgb "blue" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:6 w l lw 3 lc rgb "red" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:8 w l lw 3 lc rgb "orange" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:10 w l lw 3 lc rgb "purple" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:12 w l lw 3 lc rgb "green" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:14 w l lw 3 lc rgb "black" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:16 w l lw 3 lc rgb "magenta" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:18 w l lw 3 lc rgb "pink" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:20 w l lw 3 lc rgb "cyan" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:22 w l lw 3 lc rgb "coral" notitle "|0,0>_{K'}" ,\
  NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 2:24 w l lw 3 lc rgb "dark-violet" notitle "|0,0>_{K'}" ,\
      # TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:5 w l lw 3 lc rgb "blue" notitle "|0,1>_{K'}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:6 w l lw 3 lc rgb "green" notitle "|0,2>_{K'}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:7 w l lw 3 lc rgb "blue" notitle "|0,3>_{K'}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:8 w l lw 3 lc rgb "purple" notitle "|0,0>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:9 w l lw 3 lc rgb "red" notitle "|0,1>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:10 w l lw 3 lc rgb "orange" notitle "|0,6>_{K'}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:11 w l lw 3 lc rgb "purple" notitle "|0,4>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:12 w l lw 3 lc rgb "blue" notitle "|0,11>_{K'}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:13 w l lw 3 lc rgb "purple" notitle "|0,12>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:14 w l lw 3 lc rgb "purple" notitle "|0,13>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:15 w l lw 3 lc rgb "purple" notitle "|0,14>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:16 w l lw 3 lc rgb "purple" notitle "|0,15>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:17 w l lw 3 lc rgb "purple" notitle "|0,16>_{K}" ,\
      TNNdir . "3band_Lambda2q_dataHofstadterButterfly_q_97_MoS2_GGA_G.dat" u 2:18 w l lw 3 lc rgb "purple" notitle "|0,18>_{K}" ,\
      # plot NNdir . "3band_Lambda2q_dataHofstadterButterfly_q_797_MoS2_GGA_G.dat" u 1:3 with points pt 7 ps 0.3 lc rgb "black" nonotitle
