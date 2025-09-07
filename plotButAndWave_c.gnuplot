set terminal qt size 700,900 enhanced

set multiplot

NNdir = "./Thu-09-04/NN/"
TNNdir = "./Wed-09-03/TNN/"

waveDir = "./Sun-09-07/NN/"

set datafile separator ","
set xrange [10:80]
set yrange [*:1.69]

set key font "CMU Serif, 20"


set lmargin 14
set rmargin 13
set bmargin 4
set tics nomirror out
set xtics 20 font "CMU Serif,20"
set ytics font "CMU Serif,20"
set mytics 2
set mxtics 2
set label "(a)" at 15,1.687 font "CMU Serif, 28"
set xlabel "B(T)" font "CMU Serif,20"
set ylabel "Energy (eV)" offset -3,0 font "CMU Serif,20"

set size 0.5,1
set origin 0.0,0.0
set label "|0,0>_{K}"  at 83,1.5930  tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,1>_{K}"  at 83,1.6125  tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,0>_{K'}" at 83,1.6235  tc rgb "#ea801c" font "CMU Serif, 20"
set label "|0,2>_{K}"  at 83,1.6325  tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,1>_{K'}" at 83,1.6425  tc rgb "#ea801c" font "CMU Serif, 20"
set label "|0,3>_{K}"  at 83,1.6515  tc rgb "#1a70bb" font "CMU Serif, 20"
set label "|0,2>_{K'}" at 83,1.6619  tc rgb "#ea801c" font "CMU Serif, 20"
set label "|0,3>_{K'}" at 83,1.6809  tc rgb "#ea801c" font "CMU Serif, 20"

plot for [i=2:80] NNdir . "But_q_298_MoS2_GGA.dat" u 1:i with points pt 7 ps 0.5 lc rgb "#bdbdbd" notitle ,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:2  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:4  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:6  with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:8  with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:10 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:12 with lines lw 2  lc rgb "#1a70bb" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:14 with lines lw 2  lc rgb "#ea801c" notitle,\
  NNdir . "But_q_298_MoS2_GGA.dat" u 1:18 with lines lw 2  lc rgb "#ea801c" notitle

unset label
unset xlabel
unset ylabel
unset tics
unset title
set xrange [225:375]
set yrange [1.5925:1.69]
set label "(b)" at 230,1.687 font "CMU Serif, 28"
set xlabel "|0>" offset 0,-2 font "CMU Serif,20"
set ylabel "|{/Symbol y}|^{2}" font "CMU Serif,20"
set size 0.45,1
set origin 0.45,0.0


plot \
    NNdir . "WaveFunction_q_297_MoS2_GGA.dat" \
        using 1:( $2/8  + 1.5930) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $5/8  + 1.6145) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $6/8  + 1.6250) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $8/8  + 1.6335) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $10/8 + 1.6435) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $12/8 + 1.6515) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $15/8 + 1.6619) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $18/8 + 1.6815) w l lw 3 lc "#ea801c" notitle

unset xrange
unset yrange
unset label
unset xlabel
unset ylabel
unset tics
unset title
set xrange [225:375]
set yrange [1.5925:1.69]
set label "(c)" at 230,1.687 font "CMU Serif, 28"
set xlabel "|2>" offset 0,-2 font "CMU Serif,20"
set size 0.45,1
set origin 0.65,0.0


plot \
    NNdir . "WaveFunction_q_297_MoS2_GGA.dat" \
        using 1:( $42*1 + 1.5930) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $45*1 + 1.6145) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $46/5 + 1.6250) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $48/2 + 1.6335) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $50/5 + 1.6435) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $52/2 + 1.6515) w l lw 3 lc "#1a70bb" notitle,\
    ''  using 1:( $55/5 + 1.6619) w l lw 3 lc "#ea801c" notitle,\
    ''  using 1:( $58/5 + 1.6815) w l lw 3 lc "#ea801c" notitle


unset multiplot
