set terminal qt size 700,900 enhanced

set multiplot layout 1,2

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



plot for [i=2:80] NNdir . "But_q_297_MoS2_LDA.dat" u 1:i with points pt 7 ps 0.5 lc rgb "#bdbdbd" notitle ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:2  with lines lw 2  lc rgb "#1a70bb" title "|0,0>_{K'}" at end ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:4  with lines lw 2  lc rgb "#1a70bb" title "|0,1>_{K'}" at end ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:6  with lines lw 2  lc rgb "#ea801c" title "|0,0>_{K}" at end ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:8  with lines lw 2  lc rgb "#1a70bb" title "|0,2>_{K'}" at end ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:10 with lines lw 2  lc rgb "#ea801c" title "|0,1>_{K}" at end ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:12 with lines lw 2  lc rgb "#1a70bb" title "|0,3>_{K'}" at end ,\
  NNdir . "But_q_297_MoS2_LDA.dat" u 1:14 with lines lw 2  lc rgb "#ea801c" title "|0,2>_{K}" at end

unset xlabel
unset ylabel
unset tics
unset title
set xrange [225:375]
set xlabel "|0>" offset 0,-2 font "CMU Serif,20"
set y2label "|{/CMU Serif y}|^{2}" font "CMU Serif,20"

plot \
    NNdir . "WaveFunction_q_297_MoS2_GGA.dat" \
        using 1:( $3/5 + 1.5857581221263746)    w l lw 3 lc "#1a70bb" notitle ,\
    ''  using 1:( $5/5 + 1.6128998889188293)     w l lw 3 lc "#1a70bb"      notitle,\
    ''  using 1:( $6/5 + 1.62636) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $9/5 + 1.6385878397858133) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $11/5 +1.6521823253449328) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $13/5 +1.6629376027272889) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $14/5 +1.6771735536318174) w l lw 3 lc "#ea801c" notitle

unset multiplot
