set terminal qt size 700,900 enhanced

set multiplot

NNdir = "./Thu-09-04/NN/"
TNNdir = "./Wed-09-03/TNN/"
set datafile separator ","
# set xrange [0:51]
set yrange [*:1.687]

set key font "CMU Serif, 20"


set lmargin 13
set rmargin 13
set bmargin 4
set tics nomirror out
set xtics 10 font "CMU Serif,20"
set ytics font "CMU Serif,20"
set mytics 2
set mxtics 2
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "Energy (eV)" offset -2,0 font "CMU Serif,20"

set size 0.5,1
set origin 0.0,0.0
set label "|0,0>_{K}" at 50,1.5921               tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,1>_{K}" at 50,1.61728998889188293  tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,0>_{K'}" at 50,1.629636            tc rgb "#ea801c" font "CMU Serif, 20"
set label "|0,2>_{K}" at 50,1.6410878397858133   tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,1>_{K'}" at 50,1.6545878397858133  tc rgb "#ea801c" font "CMU Serif, 20"
set label "|0,3>_{K}" at 50,1.6645878397858133   tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,2>_{K'}" at 50,1.67755878397858133 tc rgb "#ea801c" font "CMU Serif, 20"

plot for [i=2:80] NNdir . "But_q_297_MoS2_GGA.dat" u 1:i with points pt 7 ps 0.5 lc rgb "#bdbdbd" notitle ,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:2  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:4  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:6  with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:8  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:10 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:12 with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_297_MoS2_GGA.dat" u 1:14 with lines lw 2  lc rgb "#ea801c" notitle

unset label
unset xlabel
unset ylabel
unset tics
unset title
set xrange [225:375]
set xlabel "|0>" offset 0,-2 font "CMU Serif,20"
set ylabel "|{/Symbol y}|^{2}" font "CMU Serif,20"

set size 0.5,1
set origin 0.5,0.0


plot \
    NNdir . "WaveFunction_q_297_MoS2_GGA.dat" \
        using 1:( $2/5 + 1.5857581221263746)    w l lw 3 lc "#1a70bb" notitle ,\
    ''  using 1:( $5/5 + 1.6128998889188293)     w l lw 3 lc "#1a70bb"      notitle,\
    ''  using 1:( $6/5 + 1.62636) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $8/5 + 1.6385878397858133) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $10/5 +1.6521823253449328) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $12/5 +1.6629376027272889) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $15/5 +1.6771735536318174) w l lw 3 lc "#ea801c" notitle

unset multiplot
